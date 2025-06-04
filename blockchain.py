# File: blockchain.py

import hashlib
import json
import time

class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        """
        index: int (block number)
        timestamp: float (UNIX timestamp)
        data: dict (e.g., {"image_hash": "...", "filename": "...", "label": "...", "timestamp": ...})
        previous_hash: str (hex string of previous block’s hash)
        nonce: int (for proof‐of‐work)
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Compute SHA‐256 hash over the block’s contents:
        index, timestamp, data (as JSON), previous_hash, nonce.
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()

        return hashlib.sha256(block_string).hexdigest()

    def mine(self, difficulty):
        """
        Simple Proof‐of‐Work: find a nonce so that
        hash starts with '0' * difficulty.
        """
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()


class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = []
        self.difficulty = difficulty
        # Create genesis block with index 0 and dummy data:
        genesis_block = Block(0, time.time(), {"message": "genesis"}, "0")
        genesis_block.mine(self.difficulty)
        self.chain.append(genesis_block)

    def get_last_block(self):
        return self.chain[-1]

    def add_block(self, data_dict):
        """
        data_dict: a dict containing keys:
          - "image_hash": <sha256 string>
          - "filename": <filename string>
          - "label": "genuine" or "counterfeit"
          - "timestamp": <float>
        """
        last_block = self.get_last_block()
        new_index = last_block.index + 1
        new_block = Block(new_index, time.time(), data_dict, last_block.hash)
        new_block.mine(self.difficulty)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """
        Verify that each block’s hash matches its computed hash,
        and each block’s previous_hash matches the prior block’s hash.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]
            if current.hash != current.compute_hash():
                return False
            if current.previous_hash != prev.hash:
                return False
        return True


# Quick sanity check if run directly
if __name__ == "__main__":
    bc = Blockchain(difficulty=2)
    print(f"Genesis block hash: {bc.chain[0].hash}")
    bc.add_block({
        "image_hash": "abc123",
        "filename": "test.jpg",
        "label": "genuine",
        "timestamp": time.time()
    })
    print(f"Block 1 hash: {bc.chain[1].hash}")
    print("Chain valid?", bc.is_chain_valid())
