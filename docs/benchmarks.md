# Supported Benchmarks

EGRefine supports four Text-to-SQL benchmarks. This document describes how
to obtain each, the expected on-disk layout, and the corresponding
`--benchmark` flag.

## BIRD (primary)

- **Source:** https://bird-bench.github.io/
- **Size:** 1534 dev queries, 11 databases
- **Use cases:** general schema refinement, primary results table

### Layout

```
/path/to/BIRD/
└── dev_20240627/
    ├── dev.json
    └── dev_databases/
        ├── california_schools/
        │   ├── california_schools.sqlite
        │   └── database_description/
        ├── formula_1/
        │   └── formula_1.sqlite
        └── ... (9 more)
```

### Config

```yaml
data:
  bird:
    path: "/path/to/BIRD/dev_20240627"
    split: "dev"
```

### CLI flag

```bash
egrefine-refine --benchmark bird --dbs formula_1
```

## Dr.Spider-Abbr (main schema-ambiguity benchmark)

- **Source:** https://github.com/awslabs/diagnostic-robustness-text-to-sql
- **Size:** 2853 queries, 90 databases (20 base × ~5 abbreviation variants each)
- **Use cases:** main schema-ambiguity experiment

### Layout

```
/path/to/dr_spider/DB_schema_abbreviation/
├── databases/
└── perturbations/
    ├── DB_schema_abbreviation_post.json
    └── DB_schema_abbreviation_pre.json    # original Spider data, used as upper bound
```

### Config

```yaml
data:
  drspider_abbr:
    path: "/path/to/dr_spider/DB_schema_abbreviation"
```

### CLI flags

```bash
egrefine-refine --benchmark drspider_abbr           # perturbed schema
egrefine-refine --benchmark drspider_abbr_pre       # original Spider (upper bound)
```

## Dr.Spider-Syn (synonym perturbation)

- **Source:** Same as above; adjacent directory `DB_schema_synonym/`
- **Size:** 2619 queries, 92 databases
- **Use cases:** supplementary schema-synonym experiment

### Config

```yaml
data:
  drspider_syn:
    path: "/path/to/dr_spider/DB_schema_synonym"
```

### CLI flag

```bash
egrefine-refine --benchmark drspider_syn
```

## BEAVER-NW (enterprise-grade)

- **Source:** https://huggingface.co/datasets/peterbaile/beaver
- **Size:** 88 queries, 5 databases (NW split)
- **Use cases:** enterprise-grade hard cases

### Setup

BEAVER ships as MySQL dumps, so a local MySQL server is required:

```bash
# Install MySQL (macOS via Homebrew):
brew install mysql
brew services start mysql

# Load BEAVER dumps:
mysql -u root < /path/to/BEAVER/nw/load.sql
```

### Config

```yaml
data:
  beaver:
    path: "/path/to/BEAVER"
    split: "nw"
    mysql:
      host: "localhost"
      user: "root"
      password: ""
      port: 3306
```

### CLI flag

```bash
egrefine-refine --benchmark beaver
```

## Choosing a benchmark for your use case

| Use case | Recommended benchmark |
|---|---|
| Quick demo / first-time user | BIRD `formula_1` |
| Schema ambiguity research | Dr.Spider-Abbr |
| Enterprise schema robustness | BEAVER-NW |
| General Text-to-SQL accuracy | BIRD (full) |
