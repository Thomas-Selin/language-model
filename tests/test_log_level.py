#!/usr/bin/env python3
"""
Test script to demonstrate dynamic log level adjustment
"""

import sys
import os
import logging

from language_model.helpers import configure_colored_logging, update_log_level

# Initialize logging
configure_colored_logging("DEBUG")

print("=== Testing Dynamic Log Level Changes ===")

# Test with different log levels
logging.debug("This is a DEBUG message (should be visible initially)")
logging.info("This is an INFO message (should be visible)")
logging.warning("This is a WARNING message (should be visible)")

# Change log level to INFO
print("\n--- Changing log level to INFO ---")
update_log_level("INFO")

logging.debug("This DEBUG message should NOT be visible now")
logging.info("This INFO message should still be visible")
logging.warning("This WARNING message should still be visible")

# Change log level back to DEBUG
print("\n--- Changing log level back to DEBUG ---")
update_log_level("DEBUG")

logging.debug("This DEBUG message should be visible again")
logging.info("This INFO message should be visible")

# Test invalid log level
print("\n--- Testing invalid log level ---")
update_log_level("INVALID")

print("\n=== Test Complete ===")
