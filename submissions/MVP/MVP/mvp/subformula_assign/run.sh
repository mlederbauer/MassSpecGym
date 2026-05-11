# SPEC_FILES="../data/sample/data.tsv"
# OUTPUT_DIR="/data/sample/subformulae"
# MAX_FORMULAE=60
# LABELS_FILE="../data/sample/data.tsv"

# python assign_subformulae.py --spec-files $SPEC_FILES --output-dir $OUTPUT_DIR --max-formulae $MAX_FORMULAE --labels-file $LABELS_FILE


SPEC_FILES="/data/yzhouc01/cancer/data.tsv"
OUTPUT_DIR="/data/yzhouc01/cancer/subformulae"
MAX_FORMULAE=60
LABELS_FILE="/data/yzhouc01/cancer/data.tsv"

python assign_subformulae.py --spec-files $SPEC_FILES --output-dir $OUTPUT_DIR --max-formulae $MAX_FORMULAE --labels-file $LABELS_FILE