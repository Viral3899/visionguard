"""Test configuration: ensure tests run without admin password requirement."""
import os
import sys

# Clear admin password and session secret for testing BEFORE any app imports
os.environ['HOST'] = '127.0.0.1'
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

# Also ensure we use a test database path
os.environ['TEST_DATABASE'] = '1'

# Import after env is set
# This ensures all app modules see the test configuration