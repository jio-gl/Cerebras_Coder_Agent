#!/bin/bash

# Cerebras Coding Agent Paper Compilation Script
# This script compiles the LaTeX document with proper bibliography handling

echo "🔧 Compiling Cerebras Coding Agent Paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install a LaTeX distribution."
    echo "   macOS: brew install --cask mactex"
    echo "   Ubuntu/Debian: sudo apt-get install texlive-full"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "❌ Error: bibtex not found. Please install a complete LaTeX distribution."
    exit 1
fi

# Main document name (without extension)
DOC="cerebras_coding_agent"

echo "📄 First LaTeX pass..."
pdflatex -interaction=nonstopmode "$DOC.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error in first LaTeX pass. Check the log file for details."
    exit 1
fi

echo "📚 Processing bibliography..."
bibtex "$DOC"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Bibliography processing had issues. Continuing..."
fi

echo "📄 Second LaTeX pass..."
pdflatex -interaction=nonstopmode "$DOC.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error in second LaTeX pass. Check the log file for details."
    exit 1
fi

echo "📄 Final LaTeX pass..."
pdflatex -interaction=nonstopmode "$DOC.tex"

if [ $? -ne 0 ]; then
    echo "❌ Error in final LaTeX pass. Check the log file for details."
    exit 1
fi

# Clean up auxiliary files (optional)
echo "🧹 Cleaning up auxiliary files..."
rm -f "$DOC.aux" "$DOC.bbl" "$DOC.blg" "$DOC.log" "$DOC.out" "$DOC.toc" "$DOC.fdb_latexmk" "$DOC.fls"

echo "✅ Compilation successful! Output: $DOC.pdf"
echo "📊 File size: $(du -h "$DOC.pdf" | cut -f1)"

# Check if the PDF was created and is not empty
if [ -f "$DOC.pdf" ] && [ -s "$DOC.pdf" ]; then
    echo "🎉 Paper ready for submission!"
    echo ""
    echo "📤 Sharing options:"
    echo "   • Upload to Google Drive and share the link"
    echo "   • Upload to Dropbox and create a shareable link"
    echo "   • Create a GitHub release and attach the PDF"
    echo "   • Submit to arXiv for preprint publication"
else
    echo "❌ Error: PDF was not created or is empty."
    exit 1
fi 