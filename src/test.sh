#!/bin/bash
set -e

echo "🚀 Starting Data Versioning Tests"

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
else
    echo "Running locally"
fi

# Install test dependencies if not in Docker
if ! command -v pytest &> /dev/null; then
    echo "📦 Installing test dependencies..."
    pipx install pytest pytest-cov pytest-mock pandas google-cloud-storage
fi

echo "🧪 Running tests with coverage..."
python3 -m pytest tests/test_data_versioning.py -v \
    --cov=./ \
    --cov-report=term-missing \
    --cov-report=html:coverage_report

# Check test status
if [ $? -eq 0 ]; then
    echo "✅ All tests passed successfully!"
else
    echo "❌ Some tests failed!"
    exit 1
fi

# Show coverage report location
echo "📊 Coverage report generated in coverage_report/index.html"