## Geological Model Grammar Map YAML Configuration

### File Overview
This YAML file is designed to configure and manage the vocabulary and structural grammar for the generation of geological models. It consists of two primary sections: `vocab` and `grammar`. The two combined form a probabilistic decision tree for selecting and populating geological histories formed of GeoWords.

`grammar` provides a general set of geological scenarios to sample from. They are basic geological sentence structures formed of 'geological grammar', similar to Subject Verb Object or Noun Adverb Verb Noun could be use to specify general language structures. 

The `vocab` dictionary is a weighted map between the geological grammar keys and the geological features that are represented by GeoWords. 

The randomized selection process probabilistically selects a grammar structure, then for each grammar key in the structure, selects a GeoWord from the associated list in the vocab section.

### Sections Description

#### 1. Vocab
The `vocab` section maps grammatical keys to lists of GeoWords along with their associated probability weights. This setup determines the likelihood of selecting specific geological features during the model generation process.

**Format:**
```yaml
vocab:
  <GrammarKey>:
    - [GeoWordClassName, ProbabilityWeight]
    - [GeoWordClassName, ProbabilityWeight]
    ...

; Example of two grammar keys with associated GeoWords
vocab:
  Basement:
    - [Bedrock, 0.5]
    - [InfiniteSediment, 0.5]
  Sediment:
    - [FineRepeatSediment, 0.3]
    - [CoarseRepeatSediment, 0.3]
    - [SingleRandSediment, 0.4]
```

### Details:

- **<GrammarKey>**: A category label used to classify GeoWords, such as 'Sediment' or 'Noise'.
- **GeoWordClassName**: The name of the class representing the geological feature. This must be a valid class listed in the geowords module.
- **ProbabilityWeight**: A floating-point number representing the likelihood of this GeoWord being chosen relative to others within the same category.

### 2. Grammar Lists
The `grammar` section defines various combinations of GrammarKeys that can occur together, categorized by their commonality (e.g., common, less common, rare). Each category has a set weight, defining how frequently combinations from that category should be selected.

#### Format:
```yaml
grammar:
  <FrequencyCategory>:
    weight: <WeightValue>
    structures:
      - [<GrammarKey1>, <GrammarKey2>, ...]
      - [<GrammarKey1>, <GrammarKey2>, ...]
      ...

; Example of grammar structures with associated weights
grammar:
  Common:
    weight: 0.6
    structures:
      - [Basement, Sediment, Fold]
      - [Basement, Sediment, Fault]
  LessCommon:
    weight: 0.3
    structures:
      - [Basement, Sediment, Fold, Fault]
      - [Basement, Sediment, Fold, Fault, Intrusion]
  Rare:
    weight: 0.1
    structures:
      - [Basement, Sediment, Fold, Fault, Intrusion, Fault, Erosion]
      - [Basement, Sediment, Fold, Fault, Intrusion, Uplift, Erosion]
```

#### Details:
- `<FrequencyCategory>`: A label describing the occurrence rate of the structures within this category, such as 'common'.
- `<WeightValue>`: A floating-point number that determines how often to select from this category, relative to other categories.
- `<GrammarKey>`: A reference to one of the keys defined in the `vocab` section, used to generate a structure.

### Usage:
This YAML configuration file is used by the GeologicalModelGrammar class to generate geological models based on the defined vocabulary and grammar lists. By adjusting the weights and structures in this file, users can control the variety and complexity of the generated models.