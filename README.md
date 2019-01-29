Vertical Search Blending Dataset
================================

A small sample of 25000 records from the full dataset is available in [sample.tsv](./sample.tsv). The full dataset contains 84172160 records.

The data are in tab separated values format with columns defined in [header.tsv](./header.tsv).


### Structure of the Data

Each blended Search Engine Result Page (SERP) is represented as a single line. One line has 63 fields separated by tab, describing both the query and 14 positions in the SERP.

#### Query and related descriptors

The first seven fields are common for the whole SERP:
* unique SERP id,
* hashed query id,
* number of tokens in the original query,
* number of elements above the SERP (offset) (these are not subject of blending),
* timestamp,
* available actions – list of verticals available for the given query other than organic search (separated by space),
* hardware – detected user device (`desktop`, `phone`, or `tablet`).

#### SERP positions

Each four consecutive fields of the remaining 56 fields describe one of the 14 SERP positions:

* click – 1 if the position was clicked and there was at least one subsequent click in the SERP, 2 if the position was clicked and there was no subsequent click in the SERP, and 0 if the position was not clicked (note a record can contain clicks and no last click – that means the last clicked element was not subject to blending, e.g. it was a related query link),
* propensity – the probability with which the action was taken by the logging policy,
* vertical – the numerical ID (1..20) of the vertical source chosen (0 for an organic search result),
* domain – hashed second level domain for an organic result, empty string for a non-organic vertical.
