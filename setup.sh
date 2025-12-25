#!/bin/bash
# Lilac + Eval Protocol Setup Script
# Run: ./setup.sh

set -e

echo "🌸 Setting up Lilac + Eval Protocol environment..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip -q 2>/dev/null

# Install eval-protocol with langfuse
echo "📦 Installing eval-protocol[langfuse]..."
pip install 'eval-protocol[langfuse]' -q 2>/dev/null

# Install lilac without deps (to avoid conflicts)
echo "📦 Installing lilac..."
pip install lilac --no-deps -q 2>/dev/null

# Install lilac's required dependencies
echo "📦 Installing lilac dependencies..."
pip install pandas duckdb pyarrow datasets modal cloudpickle orjson pillow tenacity itsdangerous instructor loky authlib -q 2>/dev/null

# Install clustering dependencies
echo "📦 Installing clustering dependencies..."
pip install umap-learn hdbscan hnswlib sentence-transformers -q 2>/dev/null

# Verify installation
echo ""
echo "🔍 Verifying installation..."
if python -c "import lilac; import eval_protocol" 2>/dev/null; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation failed. Please check errors above."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "  1. Copy env.template to .env and add your API keys:"
echo "     cp env.template .env"
echo ""
echo "  2. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  3. Run the example:"
echo "     pytest test_lilac_preprocessing.py -v -s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
