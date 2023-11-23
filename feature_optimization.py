import numpy as np
import math

from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pygad
import pickle

import indices

class FeatureOptimizer:
    def __init__(self, 
                 featureEncoder,
                 max_feature_count: int,
                 informativeness_func, 
                 independency_func, 
                 optimization_method: str = "genetic",
                 optimizer_args: dict = {},
                 informativeness_threshold: float = 0,
                 independency_threshold: float = 0,
                 set_independency='default'):
        self.indicesEncoder = featureEncoder #indices.IndicesClassEncoder(feature_classes, band_count, const_features)
        self.informativeness_func = informativeness_func
        self.independency_func = independency_func
        self.max_feature_count = max_feature_count
        self.optimization_method = optimization_method
        self.optimizer_args = optimizer_args
        self.informativeness_threshold = informativeness_threshold
        self.independency_threshold = independency_threshold
        self.set_independency = set_independency

    def fit(self, informativeness_data, independency_data, restoreCache = True, optimize = True):
        if restoreCache:
            self.informativeness_cache = {}
            self.independency_cache = {}

        self.informativeness_data = informativeness_data
        self.independency_data = independency_data

        self.selected_features = []
        if not optimize:
            return
        
        if self.optimization_method == "genetic":
            self.private_optimize_genetic()

        self.selected_features = self.private_threshold_dimensionality_reduction(self.selected_features)

    def transform_series(self, data, insertSelf = False):
        images = []
        if isinstance(data, list):
            images = data
        else:
            images = [data]

        shape = list(images[0].shape)
        layers = len(self.selected_features) + (0 if not insertSelf else shape[0])
        shape[0] = layers * len(images)

        transformed = np.empty(tuple(shape))
        for img_i in range(len(images)):
            img = images[img_i]
            for i in range(len(self.selected_features)):
                feature = self.indicesEncoder.getIndex(self.selected_features[i])
                transformed[i + img_i * layers] = feature.getValue(img)

            if insertSelf:
                for i in range(layers - len(self.selected_features)):
                    feature = indices.B([i])
                    transformed[len(self.selected_features) + i + img_i * layers] = feature.getValue(img)


        return transformed

    def get_fitness_(self):
        return self.private_informativeness(self.selected_features)

    def get_independency_(self):
        matrix = np.empty((len(self.selected_features), len(self.selected_features)))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.private_get_independency(self.selected_features[i], self.selected_features[j])

        return matrix
    
    def save_to(self, path):
        file_to_store = open(path, "wb")
        pickle.dump(self, file_to_store)
        file_to_store.close()

    @staticmethod
    def load_from(path):
        file_to_read = open(path, "rb")
        loaded_object = pickle.load(file_to_read)
        file_to_read.close()
        return loaded_object
    
    def private_get_informativeness(self, feature_id: int) -> float:
        if feature_id in self.informativeness_cache:
            return self.informativeness_cache[feature_id]
        
        feature = self.indicesEncoder.getIndex(feature_id)
        value_class_0 = feature.getValue(self.informativeness_data[0])
        value_class_1 = feature.getValue(self.informativeness_data[1])

        inform = self.informativeness_func(value_class_0, value_class_1)
        inform = inform if inform > self.informativeness_threshold else 0
        self.informativeness_cache[feature_id] = inform

        return inform
    
    def private_get_independency(self, feature_id_a: int, feature_id_b) -> float:
        if (feature_id_a, feature_id_b) in self.independency_cache:
            return self.independency_cache[(feature_id_a, feature_id_b)]
        
        feature_a = self.indicesEncoder.getIndex(feature_id_a)
        feature_b = self.indicesEncoder.getIndex(feature_id_b)

        value_feature_a = feature_a.getValue(self.independency_data)
        value_feature_b = feature_b.getValue(self.independency_data)

        dep = self.independency_func(value_feature_a, value_feature_b)
        dep = dep if dep > self.independency_threshold else 0

        self.independency_cache[(feature_id_a, feature_id_b)] = dep
        self.independency_cache[(feature_id_b, feature_id_a)] = dep

        return dep
    
    def private_get_independency_until(self, solution, i) -> float:
        if i == 0:
            return 1

        indep = 1
        if self.set_independency == 'default':
            for k in range(i):
                indep *= self.private_get_independency(solution[i], solution[k])
        elif self.set_independency == 'geometric_mean':
            for k in range(i):
                indep *= self.private_get_independency(solution[i], solution[k])

            indep = math.pow(indep, 1.0 / i)
        elif self.set_independency == 'harmonic_mean':
            harmonic = 0
            for k in range(i):
                harmonic += 1.0 / self.private_get_independency(solution[i], solution[k])

            indep = i / harmonic
        elif self.set_independency == 'min':
            indep = np.min([self.private_get_independency(solution[i], solution[k]) for k in range(i)])
        elif self.set_independency == 'mean':
            indep = np.mean([self.private_get_independency(solution[i], solution[k]) for k in range(i)])
        elif self.set_independency == 'weighted_harmonic_mean':
            weight = 0
            harmonic = 0
            for k in range(i):
                inform = self.private_get_informativeness(solution[i])
                weight += inform
                harmonic += inform / self.private_get_independency(solution[i], solution[k])

            indep = weight / harmonic
        
        return indep


    def private_informativeness(self, solution) -> float:
        solution = list(solution)
        solution = self.private_threshold_dimensionality_reduction(solution)

        inform = 0
        for i in range(len(solution)):
            inform += self.private_get_informativeness(solution[i]) * self.private_get_independency_until(solution, i)

        return inform    

    def private_informativeness_genetic(self, ga, solution, solution_idx) -> float:
        return self.private_informativeness(solution)

    def private_optimize_genetic(self):
        ga_opt = pygad.GA(fitness_func = self.private_informativeness_genetic,
                          gene_type=int,
                          num_genes=self.max_feature_count,
                          init_range_low=0,
                          init_range_high=self.indicesEncoder.total_length,
                          allow_duplicate_genes=True,
                          on_generation=lambda ga: print(f"\rFitness (Gen {ga.generations_completed}): {self.private_informativeness(ga.best_solution()[0])}", end=""),
                          **self.optimizer_args)
        
        ga_opt.run()
        print()

        self.selected_features = list(ga_opt.best_solution()[0])

    def private_threshold_dimensionality_reduction(self, dirty_solution):
        inform = [0]*len(dirty_solution)
        for i in range(len(dirty_solution)):
            inform[i] = self.private_get_informativeness(dirty_solution[i])

        zipped = zip(dirty_solution, inform)
        zipped = list(zipped)
        res = sorted(zipped, key = lambda x: x[1], reverse=True)
        res = list(zip(*res))

        dirty_solution = res[0]
        inform = res[1]

        counter = 0
        solution = [None]*len(dirty_solution)
        for feature in dirty_solution:
            if self.private_get_informativeness(feature) < self.informativeness_threshold:
                continue

            reject = False
            for i in range(counter):
                if self.private_get_independency(feature, solution[i]) < self.independency_threshold:
                    reject = True
                    break

            if not reject:
                solution[counter] = feature
                counter += 1

        return solution[:counter]