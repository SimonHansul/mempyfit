## v0.2.5dev1

- Fixed database error when repeatedly running pyABC backend

## v0.2.5dev2

- Updated pyABC report method
- Updated `Dataset.plot()` method to avoid duplicate legends

## v0.2.6

- Added optional kwarg `grouping_vars` to `data.add()`. This is a column index that is used to split up data into treatment combinations. 
- Added `nll_multinomial` error model.


## v0.2.7

- `grouping_vars` is expected as an index, since the previous version did not take multiple grouping  vars into account anyway. In case multiple groupings are present (e.g. food + temperature), the current solution is to include a helper column that encodes the grouping combination.