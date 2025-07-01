# Cerebras Coding Agent - Scientific Paper

This directory contains the LaTeX source files for the scientific paper on the AI-powered coding agent leveraging Cerebras hardware.

## Files

- `cerebras_coding_agent.tex` - Main LaTeX document
- `references.bib` - Bibliography file
- `figures/` - Directory containing TikZ diagrams
  - `system_architecture.tex` - System architecture diagram
  - `tool_execution_flow.tex` - Tool execution flow diagram
  - `self_improvement_cycle.tex` - Self-improvement cycle diagram
- `compile.sh` - Compilation script

## Compilation

### Local Compilation

1. Ensure you have a LaTeX distribution installed (TeX Live, MiKTeX, etc.)
2. Make the compilation script executable:
   ```bash
   chmod +x compile.sh
   ```
3. Run the compilation script:
   ```bash
   ./compile.sh
   ```

### Manual Compilation

If you prefer to compile manually:

```bash
pdflatex cerebras_coding_agent.tex
bibtex cerebras_coding_agent
pdflatex cerebras_coding_agent.tex
pdflatex cerebras_coding_agent.tex
```

### Overleaf Compilation

1. Upload all files to Overleaf
2. Set the main document to `cerebras_coding_agent.tex`
3. Compile using the "Recompile" button
4. The bibliography will be automatically processed

## Requirements

The document uses the following LaTeX packages:
- Standard packages: `amsmath`, `graphicx`, `hyperref`, `geometry`
- TikZ for diagrams: `tikz`, `pgfplots`
- Algorithm formatting: `algorithm`, `algorithmic`
- Table formatting: `booktabs`
- Code listings: `listings`, `xcolor`
- Theorem environments: `thmtools`

## Output

The compilation will generate:
- `cerebras_coding_agent.pdf` - The final paper
- Various auxiliary files (`.aux`, `.log`, `.bbl`, `.blg`)

## Paper Structure

The paper includes:
1. Abstract and Introduction
2. Mathematical Framework
3. System Architecture
4. Implementation Details
5. Self-Improvement Mechanisms
6. Empirical Results
7. Theoretical Analysis
8. Discussion and Conclusion

## Troubleshooting

- If TikZ diagrams don't compile, ensure you have the required TikZ libraries
- For bibliography issues, check that `references.bib` is in the same directory
- For missing packages, install the required LaTeX distribution packages 