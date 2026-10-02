# Input Normalization System - Production Architecture

## Overview

The Mati Input Normalization System uses a **3-layer hybrid architecture** designed for production reliability:

1. **Rule-Based Core** (80% coverage) - Fast, deterministic, bulletproof
2. **Adaptive Learning** (15% improvement) - Learns from production failures
3. **Spell Correction** (5% edge cases) - Handles typos and variations

## Architecture

```
Student Question
    ↓
[Grammar + Spell Check] (LanguageTool - 10-30ms)  ← Layer 3: Handle typos & grammar
    ↓
[Rule-Based Normalizer] (5ms)                      ← Layer 1: 80% handled here
    ↓
[Log for Learning] (async - 0ms)                   ← Layer 2: Learn from failures
    ↓
Normalized Question → RAG → Phi-1.5
```

### Layer 1: Rule-Based Core (`InputNormalizer`)

**Purpose**: Handle known patterns with zero latency

**Features**:
- 4 noise removal strategies (regex, fuzzy, learned, POS)
- Casualness to formality transformation (slang, filler words)
- Abbreviation expansion (EMF → electromotive force)
- Context expansion (implicit subjects)
- Intent classification (WHY, HOW, DESCRIBE, etc.)
- Reasoning scaffolding for Phi-1.5

**File**: `system/input_processing/input_normalizer.py`

### Layer 2: Adaptive Learning (`AdaptiveNormalizer`)

**Purpose**: Learn from production failures over time

**Features**:
- Logs every normalization with confidence scores
- Flags low-confidence cases (<0.7) for review
- Weekly pattern mining discovers new noise phrases
- Human-in-the-loop approval before adding patterns

**Files**:
- `system/input_processing/adaptive_normalizer.py` - Production wrapper
- `system/input_processing/pattern_miner.py` - Auto-discovery tool
- `scripts/run_pattern_mining.py` - Weekly review script

### Layer 3: Grammar + Spell Correction (`LanguageTool`)

**Purpose**: Handle typos, misspellings, and grammar errors

**Features**:
- **Offline** - downloads models once (~50MB), then runs locally
- 10-30ms latency (grammar + spelling in one pass)
- Handles typos ("photosintesis" → "photosynthesis")
- Fixes grammar ("what is the mitochondria" → "what are the mitochondria")
- No manual dictionary needed (auto-setup)

**Integration**: Built into `AdaptiveNormalizer`

---

## Expected Performance

### Month 1 (Rule-based only)
- Coverage: **60%**
- Accuracy: **75%**
- Manual fixes: **40%**

### Month 3 (With learning)
- Coverage: **85%**
- Accuracy: **90%**
- Manual fixes: **10%**

### Month 6 (Mature system)
- Coverage: **95%**
- Accuracy: **94%**
- Manual fixes: **2%**

---

## Usage

### Basic Usage (Production)

```python
from system.input_processing import AdaptiveNormalizer

# Initialize (loads LanguageTool + learned patterns)
normalizer = AdaptiveNormalizer(enable_spell_check=True)

# Normalize with learning
result = normalizer.normalize(
    raw_question="hey can u tell me what is fotosynthesis bro",
    user_id="student_123",
    add_scaffolding=True  # For Phi-1.5 optimization
)

print(result["clean_question"])
# Output: "What is photosynthesis?"

print(result["scaffolded_prompt"])
# Output: "Define precisely: What is photosynthesis?"
```

### Advanced: Rule-Based Only (Testing)

```python
from system.input_processing import InputNormalizer

normalizer = InputNormalizer()

result = normalizer.normalize("STATE AND EXPLAIN THE LAW OF CONSERVATION OF ENERGY")

print(result["clean_question"])
# Output: "State and explain the law of conservation of energy?"
```

### Interactive Example: Complete Flow

```python
from system.input_processing import AdaptiveNormalizer

# Initialize normalizer
normalizer = AdaptiveNormalizer(enable_spell_check=True)

# Example 1: Casual student question
raw_input = "hey bro can u explain what is photosynthesis"
result = normalizer.normalize(raw_input, user_id="student_001")

print(f"Original: {raw_input}")
print(f"Normalized: {result['clean_question']}")
print(f"Transformations: {result['transformations_applied']}")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2f}")

# Example 2: Exam-style question with fluff
raw_input = "WITH REFERENCE TO THE DIAGRAM ABOVE EXPLAIN THE LAW OF CONSERVATION"
result = normalizer.normalize(raw_input, user_id="student_002")

print(f"\nOriginal: {raw_input}")
print(f"Normalized: {result['clean_question']}")
print(f"Transformations: {result['transformations_applied']}")

# Example 3: Vague question needing context
raw_input = "what happens when switch is on"
result = normalizer.normalize(raw_input, user_id="student_003")

print(f"\nOriginal: {raw_input}")
print(f"Normalized: {result['clean_question']}")
print(f"Transformations: {result['transformations_applied']}")
```

**Expected Output:**
```
Original: hey bro can u explain what is photosynthesis
Normalized: What is photosynthesis?
Transformations: ['formalized', 'removed_noise']
Intent: DESCRIBE
Confidence: 1.00

Original: WITH REFERENCE TO THE DIAGRAM ABOVE EXPLAIN THE LAW OF CONSERVATION
Normalized: Explain the law of conservation of energy?
Transformations: ['removed_regex', 'fixed_caps']

Original: what happens when switch is on
Normalized: What happens in a electric circuit when switch is on?
Transformations: ['expanded_context']
```

---

## Weekly Pattern Mining

Run this **every week** to discover new patterns:

```bash
python scripts/run_pattern_mining.py
```

**What it does**:
1. Loads all normalization logs
2. Finds repeated patterns in low-confidence cases
3. Suggests new noise phrases
4. **Human review** - you approve/reject each suggestion
5. Auto-adds approved patterns to the normalizer

**Interactive Example Session:**
```
Loading logs from mati_data/normalization_logs/feedback_db.jsonl...
Found 1,247 normalization entries

Analyzing low-confidence cases (confidence < 0.7)...
Found 89 low-confidence cases

Mining patterns...
Discovered 15 potential noise patterns

[1/15] Found pattern:
  Phrase: "can u pls"
  Frequency: 23 times
  Confidence: 0.87
  Examples:
    - "can u pls explain mitosis"
    - "hey can u pls help me with photosynthesis"
    - "can u pls tell me about circuits"
  
  Add to noise phrases? (y/n/q): y
  Added to input_normalizer_phrases.json

[2/15] Found pattern:
  Phrase: "like basically"
  Frequency: 18 times
  Confidence: 0.75
  Examples:
    - "like basically what is DNA"
    - "so like basically how does it work"
  
  Add to noise phrases? (y/n/q): y
  Added to input_normalizer_phrases.json

[3/15] Found pattern:
  Phrase: "state and explain"
  Frequency: 45 times
  Confidence: 0.62
  Examples:
    - "state and explain the law of conservation"
    - "state and explain Newton's first law"
  
  Add to noise phrases? (y/n/q): n
  Skipped (valid instruction verb)

...

Summary:
  Reviewed: 15 patterns
  Approved: 8 patterns
  Rejected: 7 patterns
  
Report saved to: mati_data/normalization_logs/pattern_mining_report.md
```

---

## File Structure

```
system/input_processing/
├── __init__.py                    # Exports all components
├── input_normalizer.py            # Layer 1: Rule-based core
├── adaptive_normalizer.py         # Layer 2: Production wrapper
└── pattern_miner.py               # Auto-discovery tool

scripts/
└── run_pattern_mining.py          # Weekly review script

mati_data/
├── input_normalizer_phrases.json  # Learned noise phrases
└── normalization_logs/
    ├── feedback_db.jsonl          # All normalizations
    ├── low_confidence_cases.jsonl # Flagged for review
    └── pattern_mining_report.md   # Last mining results
```

---

## Setup

### Install Dependencies

```bash
# Required - Grammar + spell correction (auto-downloads models ~50MB)
pip install language-tool-python

# Optional (for POS tagging - better pattern detection)
pip install spacy
python -m spacy download en_core_web_sm
```

**Note**: LanguageTool auto-downloads models on first use. No manual setup needed.

---

## Design Principles

### Best Practices
- Keep rules **simple and explicit**
- Log **everything** for learning
- Review patterns **weekly** with human oversight
- Update patterns **gradually** (don't rush)

### Avoid
- Don't remove **valid question verbs** (e.g., "state and explain")
- Don't auto-approve patterns without review
- Don't change core logic too frequently
- Don't trust low-frequency patterns (<10 occurrences)

---

## Future Enhancements (Optional)

### Semantic Caching (MiniLM)
Cache normalized questions by semantic similarity:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# 70% cache hit rate expected
```

### Advanced Grammar Rules (LanguageTool)
Custom grammar rules for educational domain:
```python
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
# Handles complex grammar issues
```

---

## Monitoring

### Key Metrics to Track

1. **Confidence Distribution**
   - Target: >80% of queries with confidence >0.7
   
2. **Low-Confidence Rate**
   - Target: <20% of queries flagged

3. **Pattern Mining Yield**
   - Target: 5-10 new patterns per week (Month 1)
   - Target: 1-2 new patterns per week (Month 6)

### Commands

```bash
# View low-confidence cases
cat mati_data/normalization_logs/low_confidence_cases.jsonl | tail -n 10

# Count total normalizations
wc -l mati_data/normalization_logs/feedback_db.jsonl

# View last mining report
cat mati_data/normalization_logs/pattern_mining_report.md
```

---

## FAQ

**Q: Why not use a fine-tuned LLM for normalization?**
A: Too slow (50-200ms), unpredictable, and can hallucinate. Rules are deterministic and <5ms.

**Q: What if LanguageTool doesn't have a word?**
A: It skips unknown words. You can add domain-specific words to the dictionary.

**Q: How do I handle regional English variations?**
A: Add them to `slang_to_formal` dictionary in `InputNormalizer`.

**Q: Can I disable learning/logging?**
A: Yes, just use `InputNormalizer` directly instead of `AdaptiveNormalizer`.

---

```
Student Question
    ↓
[Grammar + Spell Check] (LanguageTool - 10-30ms)  ← Layer 3: Handle typos & grammar
    ↓
[Rule-Based Normalizer] (5ms)                      ← Layer 1: 80% handled here
    ↓
[Log for Learning] (async - 0ms)                   ← Layer 2: Learn from failures
    ↓
Normalized Question → RAG → Phi-1.5
```

### Layer 1: Rule-Based Core (`InputNormalizer`)

**Purpose**: Handle known patterns with zero latency

**Features**:
- ✅ 4 noise removal strategies (regex, fuzzy, learned, POS)
- ✅ Casualness → formality transformation (slang, filler words)
- ✅ Abbreviation expansion (EMF → electromotive force)
- ✅ Context expansion (implicit subjects)
- ✅ Intent classification (WHY, HOW, DESCRIBE, etc.)
- ✅ Reasoning scaffolding for Phi-1.5

**File**: `system/input_processing/input_normalizer.py`

### Layer 2: Adaptive Learning (`AdaptiveNormalizer`)

**Purpose**: Learn from production failures over time

**Features**:
- 📊 Logs every normalization with confidence scores
- 🚨 Flags low-confidence cases (<0.7) for review
- 🔄 Weekly pattern mining discovers new noise phrases
- 👤 Human-in-the-loop approval before adding patterns

**Files**:
- `system/input_processing/adaptive_normalizer.py` - Production wrapper
- `system/input_processing/pattern_miner.py` - Auto-discovery tool
- `scripts/run_pattern_mining.py` - Weekly review script

### Layer 3: Grammar + Spell Correction (`LanguageTool`)

**Purpose**: Handle typos, misspellings, and grammar errors

**Features**:
- ✅ **Offline** - downloads models once (~50MB), then runs locally
- ✅ 10-30ms latency (grammar + spelling in one pass)
- ✅ Handles typos ("photosintesis" → "photosynthesis")
- ✅ Fixes grammar ("what is the mitochondria" → "what are the mitochondria")
- ✅ No manual dictionary needed (auto-setup)

**Integration**: Built into `AdaptiveNormalizer`

---

## 📊 Expected Performance

### Month 1 (Rule-based only)
- Coverage: **60%**
- Accuracy: **75%**
- Manual fixes: **40%**

### Month 3 (With learning)
- Coverage: **85%**
- Accuracy: **90%**
- Manual fixes: **10%**

### Month 6 (Mature system)
- Coverage: **95%**
- Accuracy: **94%**
- Manual fixes: **2%**

---

## 🚀 Usage

### Basic Usage (Production)

```python
from system.input_processing import AdaptiveNormalizer

# Initialize (loads SymSpell + learned patterns)
normalizer = AdaptiveNormalizer(enable_spell_check=True)

# Normalize with learning
result = normalizer.normalize(
    raw_question="hey can u tell me what is fotosynthesis bro",
    user_id="student_123",
    add_scaffolding=True  # For Phi-1.5 optimization
)

print(result["clean_question"])
# Output: "What is photosynthesis?"

print(result["scaffolded_prompt"])
# Output: "Define precisely: What is photosynthesis?"
```

### Advanced: Rule-Based Only (Testing)

```python
from system.input_processing import InputNormalizer

normalizer = InputNormalizer()

result = normalizer.normalize("STATE AND EXPLAIN THE LAW OF CONSERVATION OF ENERGY")

print(result["clean_question"])
# Output: "State and explain the law of conservation of energy?"
```

---

## 🔄 Weekly Pattern Mining

Run this **every week** to discover new patterns:

```bash
python scripts/run_pattern_mining.py
```

**What it does**:
1. Loads all normalization logs
2. Finds repeated patterns in low-confidence cases
3. Suggests new noise phrases
4. **Human review** - you approve/reject each suggestion
5. Auto-adds approved patterns to the normalizer

**Example output**:
```
[1/15] Found pattern:
  Phrase: "can u pls"
  Frequency: 23 times
  Confidence: 0.87
  Examples:
    - "can u pls explain mitosis"
    - "hey can u pls help me"
  
  Add to noise phrases? (y/n/q): y
  ✅ Added!
```

---

## 📁 File Structure

```
system/input_processing/
├── __init__.py                    # Exports all components
├── input_normalizer.py            # Layer 1: Rule-based core
├── adaptive_normalizer.py         # Layer 2: Production wrapper
└── pattern_miner.py               # Auto-discovery tool

scripts/
└── run_pattern_mining.py          # Weekly review script

mati_data/
├── input_normalizer_phrases.json  # Learned noise phrases
└── normalization_logs/
    ├── feedback_db.jsonl          # All normalizations
    ├── low_confidence_cases.jsonl # Flagged for review
    └── pattern_mining_report.md   # Last mining results
```

---

## 🛠️ Setup

### Install Dependencies

```bash
# Required - Grammar + spell correction (auto-downloads models ~50MB)
pip install language-tool-python

# Optional (for POS tagging - better pattern detection)
pip install spacy
python -m spacy download en_core_web_sm
```

**That's it!** LanguageTool auto-downloads models on first use. No manual setup needed.

---

## 🎯 Design Principles

### ✅ DO
- Keep rules **simple and explicit**
- Log **everything** for learning
- Review patterns **weekly** with human oversight
- Update patterns **gradually** (don't rush)

### ❌ DON'T
- Don't remove **valid question verbs** (e.g., "state and explain")
- Don't auto-approve patterns without review
- Don't change core logic too frequently
- Don't trust low-frequency patterns (<10 occurrences)

---

## 🔮 Future Enhancements (Optional)

### Semantic Caching (MiniLM)
Cache normalized questions by semantic similarity:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# 70% cache hit rate expected
```

### Advanced Spell Check (LanguageTool)
Grammar + spelling in one:
```python
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
# Handles complex grammar issues
```

---

## 📈 Monitoring

### Key Metrics to Track

1. **Confidence Distribution**
   - Target: >80% of queries with confidence >0.7
   
2. **Low-Confidence Rate**
   - Target: <20% of queries flagged

3. **Pattern Mining Yield**
   - Target: 5-10 new patterns per week (Month 1)
   - Target: 1-2 new patterns per week (Month 6)

### Commands

```bash
# View low-confidence cases
cat mati_data/normalization_logs/low_confidence_cases.jsonl | tail -n 10

# Count total normalizations
wc -l mati_data/normalization_logs/feedback_db.jsonl

# View last mining report
cat mati_data/normalization_logs/pattern_mining_report.md
```

---

## 🎓 FAQ

**Q: Why not use a fine-tuned LLM for normalization?**
A: Too slow (50-200ms), unpredictable, and can hallucinate. Rules are deterministic and <5ms.

**Q: What if SymSpell doesn't have a word?**
A: It skips unknown words. You can add domain-specific words to the dictionary.

**Q: How do I handle regional English variations?**
A: Add them to `slang_to_formal` dictionary in `InputNormalizer`.

**Q: Can I disable learning/logging?**
A: Yes, just use `InputNormalizer` directly instead of `AdaptiveNormalizer`.
