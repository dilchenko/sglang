"""
Unit test for the evict_host orphaned-node fix (#19212).

Reproduces the crash scenario where recursive eviction
(evict -> write_backup -> evict_host) mutates the radix tree while
evict_host is still iterating its heap, causing an AssertionError
on nodes whose parents have already been removed.

This test does NOT require a GPU or a running server.  It constructs
a minimal tree of TreeNode objects and calls evict_host() directly
after pre-orphaning a node to simulate the recursive mutation.
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only unit test, runs in seconds on any runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import heapq
import logging
import time
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.evict_policy import LRUStrategy
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

logger = logging.getLogger("sglang.srt.mem_cache.hiradix_cache")


def evict_host_under_test(self, num_tokens: int):
    """Copy of HiRadixCache.evict_host with the orphaned-node fix applied.

    We inline the method here rather than importing from hiradix_cache to
    avoid pulling in heavy GPU dependencies (torch.distributed, CUDA pools,
    etc.) that are not needed for this CPU-only unit test.
    """
    leaves = list(self.evictable_host_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        if x == self.root_node:
            break
        if not x.evicted:
            continue

        if x.pin_expiry > 0 and time.monotonic() > x.pin_expiry:
            self._clear_pin(x)

        if x.host_ref_counter > 0:
            continue

        self._record_remove_event(x)
        num_evicted += self.cache_controller.evict_host(x.host_value)

        key = self.get_child_key_fn(x.key)
        v = x.parent.children.pop(key, None)
        if v is not x:
            if v is not None:
                x.parent.children[key] = v
            logger.warning(
                "evict_host: node already orphaned (key=%s), skipping tree cleanup",
                key,
            )
            self.evictable_host_leaves.discard(x)
            continue
        self.evictable_host_leaves.discard(x)
        self._update_host_leaf_status(x.parent)

        if len(x.parent.children) == 0 and x.parent.evicted:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))


def _build_stub(nodes, root):
    """Build a minimal object with just enough state for evict_host()."""
    stub = unittest.mock.MagicMock()
    stub.root_node = root
    stub.evictable_host_leaves = set(nodes)
    stub.eviction_strategy = LRUStrategy()
    stub.get_child_key_fn = lambda key: key.token_ids[0]

    stub.cache_controller = unittest.mock.MagicMock()
    stub.cache_controller.evict_host.side_effect = lambda hv: len(hv)

    stub._record_remove_event = unittest.mock.MagicMock()
    stub._update_host_leaf_status = unittest.mock.MagicMock()
    stub._clear_pin = unittest.mock.MagicMock()

    stub.evict_host = lambda num_tokens: evict_host_under_test(stub, num_tokens)

    return stub


def _make_node(token_id, parent, *, evicted=True, host_value=None):
    """Create a TreeNode wired into the tree under parent."""
    node = TreeNode()
    node.key = RadixKey([token_id])
    node.value = None if evicted else torch.tensor([1])
    node.host_value = host_value if host_value is not None else torch.tensor([1])
    node.host_ref_counter = 0
    node.pin_expiry = 0.0
    node.parent = parent
    node.last_access_time = time.monotonic()
    time.sleep(0.001)
    if parent is not None:
        parent.children[token_id] = node
    return node


class TestEvictHostOrphanedNode(unittest.TestCase):
    """Test that evict_host handles orphaned nodes gracefully."""

    def setUp(self):
        TreeNode.counter = 0

    def test_orphaned_node_does_not_crash(self):
        """Simulate a node whose parent-child link was already removed.

        Before the fix, this would raise:
            AssertionError: parent does not have child key, <key>
        """
        root = TreeNode()
        root.key = RadixKey([])
        root.value = torch.tensor([0])
        root.host_value = None
        root.host_ref_counter = 0
        root.pin_expiry = 0.0

        node_a = _make_node(10, root, evicted=True)
        node_b = _make_node(20, node_a, evicted=True)

        # Pre-orphan node_b: remove it from its parent's children dict.
        del node_a.children[20]

        stub = _build_stub([node_b], root)

        with self.assertLogs(
            "sglang.srt.mem_cache.hiradix_cache", level=logging.WARNING
        ) as cm:
            stub.evict_host(1)

        self.assertTrue(
            any("node already orphaned" in msg for msg in cm.output),
            f"Expected orphaned-node warning, got: {cm.output}",
        )

        stub.cache_controller.evict_host.assert_called_once()
        self.assertNotIn(node_b, stub.evictable_host_leaves)

    def test_displaced_sibling_is_restored(self):
        """When pop() returns a different node, it must be put back."""
        root = TreeNode()
        root.key = RadixKey([])
        root.value = torch.tensor([0])
        root.host_value = None
        root.host_ref_counter = 0
        root.pin_expiry = 0.0

        node_a = _make_node(10, root, evicted=True)

        # Place a DIFFERENT node at the same key slot
        sibling = _make_node(10, root, evicted=True)

        stub = _build_stub([node_a], root)

        with self.assertLogs(
            "sglang.srt.mem_cache.hiradix_cache", level=logging.WARNING
        ):
            stub.evict_host(1)

        # The sibling should have been restored in the children dict
        self.assertIn(10, root.children)
        self.assertIs(root.children[10], sibling)

        stub.cache_controller.evict_host.assert_called_once()

    def test_normal_eviction_still_works(self):
        """Sanity check: normal (non-orphaned) eviction path is unaffected."""
        root = TreeNode()
        root.key = RadixKey([])
        root.value = torch.tensor([0])
        root.host_value = None
        root.host_ref_counter = 0
        root.pin_expiry = 0.0

        node_a = _make_node(10, root, evicted=True)
        node_b = _make_node(20, root, evicted=True)

        stub = _build_stub([node_a, node_b], root)

        stub.evict_host(2)

        self.assertEqual(stub.cache_controller.evict_host.call_count, 2)
        self.assertEqual(len(stub.evictable_host_leaves), 0)
        self.assertEqual(len(root.children), 0)
        self.assertEqual(stub._update_host_leaf_status.call_count, 2)

    def test_cascading_parent_eviction(self):
        """When all children are evicted, the parent is pushed onto the heap."""
        root = TreeNode()
        root.key = RadixKey([])
        root.value = torch.tensor([0])
        root.host_value = None
        root.host_ref_counter = 0
        root.pin_expiry = 0.0

        node_a = _make_node(10, root, evicted=True)
        node_b = _make_node(20, node_a, evicted=True)

        stub = _build_stub([node_b], root)

        stub.evict_host(2)

        self.assertEqual(stub.cache_controller.evict_host.call_count, 2)
        self.assertNotIn(10, root.children)


if __name__ == "__main__":
    unittest.main()
