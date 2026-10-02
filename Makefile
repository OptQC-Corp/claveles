# Common definition of ignore patterns
LICENSE_IGNORES = -ignore "**/*.md" -ignore "**/*.txt" -ignore "**/*.bib" \
                  -ignore "**/*.json" -ignore "**/*.yaml" -ignore "**/*.yml" \
                  -ignore "**/*.toml" -ignore "**/*.lock" -ignore "**/*.ipynb" \
                  -ignore "**/*.png" -ignore "**/*.gif" -ignore "**/*.svg" \
                  -ignore "**/*.drawio" -ignore "**/*.npz" -ignore "**/*.typed" \
                  -ignore "LICENSE*" -ignore "NOTICE" -ignore "THIRD_PARTY_LICENSES.md"

.PHONY: license-add license-check
.SILENT: license-add license-check

# Add standard Apache 2.0 headers to files
license-add:
	addlicense -c "The Claveles Contributors" -l apache $(LICENSE_IGNORES) .

# Check for missing license headers (same command used in CI)
license-check:
	addlicense -check $(LICENSE_IGNORES) . && echo "✅ All license headers are up to date."