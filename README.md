# Optimized Feature Engineering for Semantic Segmentation of Satellite Imagery

This repository represents source code used during article writing.

All experiments conducted within the framework of the HORIZON Europe project "[Satellites for Wilderness Inspection and Forest Threat Tracking](https://swiftt.eu/)" (SWIFTT).

## Data avaibility

Data will be made avaible on request. For requests, please contact e-mail: mmda.ipt.kpi@gmail.com

## Dependencies

All results was conducted using Python 3.11.3. Required libraries listed in requirements.yml.

## Documentattion

### Targets of .ipynb

- create_subdivision.ipynb - used to create train/test subdivision of Dataset 1
- threshold_gridsearh.ipynb - used to conduct Experiment 1
- fintess_metric_relation.ipynb - used to plot results of Experiment 1
- config_search.ipynb - used to find feature sets and evaluate models in Experiment 2
- apply.ipynb - used to apply models from Experiment 2 on test sites
- violinplotting.ipynb - used to plot violinplots of selected features

### Feature selection

In feature_optimization.py defined class FeatureOptimizer that used to select features within classes that implements IIndex interface from indices.py.

#### Hyperparameters

- optimizer_args - hyperparameters of [PyGAD GA](https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class). Dictionary
- featureEncoder - object that encode features to integers. In indices.py implemented IndicesClassEncoder, IndicesClassEncoderEq, BakedIndiceClassEncoder from which it is recommened to use IndicesClassEncoderEq.
- max_feature_count - max number of selected features. Integer
- informativeness_func - individual informativeness function. Delegate. In feature_opt_functions.py implemented bhattacharyya_distance.
- independency_func - pairwise independence function. Delegate. In feature_opt_functions.py implemented pearson_independency, spearman_independency, distance_independency from which spearman_independency is recommened to use.
- optimization_method - type of optimized. Str. Can only be 'generic'.
- informativeness_threshold - minimal value of informativeness_func of feature from which it is not rejected.
- independency_threshold - minimal value of independency_func of feature from which it is not rejected.
- set_independency - independence function. Str. 'default' for Multiplication. 'geometric_mean', 'harmonic_mean', 'min', 'mean', 'weighted_harmonic_mean' for different Generalized mean.

#### Usage example

```python3
from feature_optimization import FeatureOptimizer
import feature_opt_functions as funcs
import indices

encoder = indices.IndicesClassEncoderEq(indices.NORMP4, list(range(1, 12)))
inform_cache = {}
indep_cache = {}

args = { 
    "num_generations":150, 
    "num_parents_mating":3,
    "parent_selection_type":"sss",
    "keep_elitism":1,
    "sol_per_pop":150,
    "mutation_probability":0.25,
    "parallel_processing":8
}

opt = FeatureOptimizer(encoder, 12,
        funcs.bhattacharyya_distance, 
        funcs.spearman_independency, 
        optimization_method="genetic",
        optimizer_args=args,
        informativeness_threshold=informativeness_threshold, 
        independency_threshold=independency_threshold,
        set_independency="geometric_mean")

opt.informativeness_cache = inform_cache
opt.independency_cache = indep_cache

# data = [negative_samples, positive_samles]
opt.fit(data, data[1], False)

# list of encoded selected features
print(opt.selected_features)
# fitness value of selected features
print(opt.get_fitness_)
# matrix of pairwise independence of selcted features
print(opt.get_independency_)
```

