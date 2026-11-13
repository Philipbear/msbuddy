"""Versioned feature, selection and absent-answer probability contracts."""
import unittest
from unittest.mock import patch

import numpy as np

from msbuddy import Adduct, Formula, MetaFeature, MsbuddyConfig, Msbuddy
from msbuddy.base import CandidateFormula
from msbuddy._calibration import CONTEXT_NAMES, calibrated_probabilities, feature_context
from msbuddy._candidate_selection import select_candidates, selector_features
from msbuddy._features import EXTRA_FEATURE_NAMES
from msbuddy.load import _load_versioned_models
from msbuddy.ml import predict_formula_probability, calc_fdr
from msbuddy.utils import read_formula


class Model:
    def __init__(self, dimensions, scores=None):
        self.dimensions = dimensions
        self.scores = scores

    def num_feature(self):
        return self.dimensions

    def predict(self, matrix, **kwargs):
        if matrix.shape[1] != self.dimensions:
            raise AssertionError('Wrong inference feature schema')
        return np.asarray(self.scores[:len(matrix)]) if self.scores is not None else np.zeros(len(matrix))


def specification():
    return {'method':'null_softmax','context_center':[0.] * len(CONTEXT_NAMES),
            'context_scale':[1.] * len(CONTEXT_NAMES),
            'coefficients':[0.] * (len(CONTEXT_NAMES) + 1),'inverse_temperature':1.}


def candidate(formula='C6H12O6', ion='[M+H]+'):
    neutral = Formula(read_formula(formula))
    adduct = Adduct(ion, ion.endswith('+'))
    charged = Formula._from_array(neutral.array * adduct.m + adduct.net_formula.array, adduct.charge)
    return CandidateFormula(neutral, charged, mz_error=0., exp_ms2_sum_int=0., db_existed=True)


class ModelBundleTests(unittest.TestCase):
    def bundle(self):
        return {'format_version':2,'feature_version':2,
                'models':{name:Model(size) for name,size in zip(['joint','isotope','fragment','mass'],[95,71,94,70])},
                'calibrations':{name:specification() for name in ['joint','isotope','fragment','mass']},
                'candidate_selector':Model(73),'candidate_limits':[50,50,75,50]}

    def test_mode_names_map_to_the_correct_evidence_models(self):
        dependencies = {}
        _load_versioned_models(self.bundle(), dependencies)
        self.assertEqual(dependencies['model_ms1_noms2'].num_feature(),71)
        self.assertEqual(dependencies['model_noms1_ms2'].num_feature(),94)
        self.assertEqual(dependencies['candidate_limits'],(50,50,75,50))

    def test_incompatible_bundle_is_rejected_atomically(self):
        value = self.bundle(); value['models']['fragment'] = Model(52)
        dependencies = {'existing':True}
        with self.assertRaisesRegex(ValueError,'feature count'):
            _load_versioned_models(value, dependencies)
        self.assertEqual(dependencies,{'existing':True})

    def test_invalid_ensemble_scale_is_rejected_at_load(self):
        value = self.bundle()
        value['models']['mass'] = {'model':Model(70), 'new_score_scale':0.}
        with self.assertRaisesRegex(ValueError, 'weights or scales'):
            _load_versioned_models(value, {})

    def test_absolute_tolerances_use_legacy_fallback(self):
        value = self.bundle()
        value['legacy_bundle'] = [Model(n,[1.]) for n in [53,52,29,28]] + [1.,0.] * 4
        dependencies = {}
        _load_versioned_models(value, dependencies)
        cf = candidate()
        feature = MetaFeature('a',cf.charged_formula.mass,1)
        feature.candidate_formula_list = [cf]
        predict_formula_probability([feature],0,1,MsbuddyConfig(ppm=False,ms1_tol=.01),dependencies)
        self.assertAlmostEqual(cf.estimated_prob,1/(1+np.exp(-1)))

    def test_mixed_charge_batch_routes_each_query_independently(self):
        value = self.bundle()
        value['legacy_bundle'] = [Model(n,[1.]) for n in [53,52,29,28]] + [1.,0.] * 4
        dependencies = {}
        _load_versioned_models(value, dependencies)
        single = candidate()
        multiple = candidate(ion='[M+2H]2+')
        features = [MetaFeature('a',single.charged_formula.mass,1),
                    MetaFeature('b',multiple.charged_formula.mass,2)]
        features[0].candidate_formula_list = [single]
        features[1].candidate_formula_list = [multiple]
        predict_formula_probability(features,0,2,MsbuddyConfig(),dependencies)
        self.assertAlmostEqual(single.estimated_prob,.5)
        self.assertAlmostEqual(multiple.estimated_prob,1/(1+np.exp(-1)))

    def test_unvalidated_adduct_uses_fallback_and_equivalent_spelling_uses_new_model(self):
        value = self.bundle()
        value['validated_adducts'] = ['[M+H]+','[M+CH2O2-H]-']
        value['legacy_bundle'] = [Model(n,[1.]) for n in [53,52,29,28]] + [1.,0.] * 4
        dependencies = {}
        _load_versioned_models(value, dependencies)
        uncommon = candidate(ion='[M+H-H2O]+')
        alias = candidate(ion='[M+HCOO]-')
        features = [MetaFeature('a',uncommon.charged_formula.mass,1,adduct='[M+H-H2O]+'),
                    MetaFeature('b',alias.charged_formula.mass,-1,adduct='[M+HCOO]-')]
        features[0].candidate_formula_list = [uncommon]
        features[1].candidate_formula_list = [alias]
        predict_formula_probability(features,0,2,MsbuddyConfig(),dependencies)
        self.assertAlmostEqual(uncommon.estimated_prob,1/(1+np.exp(-1)))
        self.assertAlmostEqual(alias.estimated_prob,.5)

    def test_new_probability_path_preserves_uncertainty_for_single_candidate(self):
        dependencies = {}
        _load_versioned_models(self.bundle(), dependencies)
        cf = candidate()
        feature = MetaFeature('a',cf.charged_formula.mass,1)
        feature.candidate_formula_list=[cf]
        predict_formula_probability([feature],0,1,MsbuddyConfig(),dependencies)
        calc_fdr([feature],0,1)
        self.assertAlmostEqual(cf.estimated_prob,.5)
        self.assertAlmostEqual(cf.estimated_fdr,.5)

    def test_null_state_and_formula_probabilities_conserve_mass(self):
        scores=np.array([2.,-1.,3.])
        probabilities,absent=calibrated_probabilities(scores,np.zeros(len(CONTEXT_NAMES)),specification())
        self.assertAlmostEqual(probabilities.sum()+absent,1.)
        np.testing.assert_array_equal(np.argsort(-scores),np.argsort(-probabilities))
        self.assertGreater(absent,0.)

    def test_calibration_isotope_context_matches_training_precision(self):
        cf = candidate()
        cf.ms1_isotope_similarity = 0.8925751969218254
        feature = MetaFeature('a',cf.charged_formula.mass,1)
        feature.ms1_processed = [1,2]
        feature.candidate_formula_list = [cf]
        context = feature_context(feature,np.array([0.]))
        self.assertEqual(context[3],float(np.float32(cf.ms1_isotope_similarity)))

    def test_invalid_calibration_scale_is_rejected(self):
        value=specification();value['context_scale'][0]=0.
        with self.assertRaises(ValueError):
            calibrated_probabilities(np.array([1.]),np.zeros(len(CONTEXT_NAMES)),value)

    def test_learned_selection_and_small_pool_ties_use_consistent_order(self):
        cands=[candidate() for _ in range(3)]
        feature=MetaFeature('a',cands[0].charged_formula.mass,1)
        dependencies={'candidate_selector':Model(73,[.2,.8,.1]),'candidate_limits':[2]*4}
        kept=select_candidates(feature,cands,True,5,10,dependencies)
        self.assertEqual(kept,[cands[1],cands[0]])
        kept=select_candidates(feature,cands,True,5,10,dependencies,limit=10)
        self.assertEqual(kept,[cands[1],cands[0],cands[2]])

    def test_halogen_identity_and_multimer_counts_are_explicit(self):
        cf=candidate('C6H10Cl2','[2M+H]+')
        feature=MetaFeature('a',cf.charged_formula.mass,1,adduct='[2M+H]+')
        matrix=selector_features(feature,[cf],True,5,10)
        self.assertEqual(matrix[0,29+EXTRA_FEATURE_NAMES.index('neutral_Cl')],2)
        self.assertEqual(matrix[0,29+EXTRA_FEATURE_NAMES.index('ion_Cl')],4)
        self.assertEqual(matrix[0,29+EXTRA_FEATURE_NAMES.index('precursor_multiplier')],2)

    def test_candidate_budget_configuration_rejects_invalid_values(self):
        for value in [True,0,-1,1.5,'50',10001]:
            with self.assertRaises(ValueError):MsbuddyConfig(max_candidates=value)
        self.assertEqual(MsbuddyConfig(max_candidates=np.int64(75)).max_candidates,75)
        self.assertIsNone(MsbuddyConfig().max_candidates)

    def test_partial_configuration_update_preserves_settings_and_models(self):
        engine = Msbuddy.__new__(Msbuddy)
        engine.config = MsbuddyConfig(halogen=True, parallel=True, timeout_secs=90,
                                     ms1_tol=8, element_ranges={'C':(1,60)})
        dependencies = {'model_ms1_ms2':object()}
        engine._dependencies = dependencies
        with patch('msbuddy.main.init_db') as initialize:
            engine.update_config(max_candidates=75)
            engine.update_config(max_candidates=50)
        initialize.assert_not_called()
        self.assertIs(engine._dependencies, dependencies)
        self.assertEqual(engine.config.ms1_tol, 8)
        self.assertEqual(engine.config.timeout_secs, 120)
        self.assertEqual(engine.config.element_ranges['C'], (1,60))
        self.assertEqual(engine.config.db_mode, 1)
        self.assertEqual(engine.config.max_candidates, 50)

    def test_configuration_update_can_enable_default_halogen_ranges(self):
        engine = Msbuddy.__new__(Msbuddy)
        engine.config = MsbuddyConfig()
        engine._dependencies = {}
        engine.update_config(halogen=True)
        self.assertGreater(engine.config.element_ranges['Cl'][1], 0)

    def test_partial_update_does_not_reuse_mutated_caller_bounds(self):
        bounds = {'C':[1,60]}
        engine = Msbuddy.__new__(Msbuddy)
        engine.config = MsbuddyConfig(element_ranges=bounds)
        engine._dependencies = {}
        bounds['C'][1] = 5
        engine.update_config(max_candidates=75)
        self.assertEqual(engine.config.element_ranges['C'], (1,60))

    def test_failed_configuration_update_keeps_existing_state(self):
        engine = Msbuddy.__new__(Msbuddy)
        engine.config = MsbuddyConfig()
        engine._dependencies = {}
        original = engine.config
        with patch('msbuddy.main.init_db', side_effect=OSError('unavailable')):
            with self.assertRaises(OSError):engine.update_config(data_dir='/unavailable')
        self.assertIs(engine.config, original)


if __name__=='__main__':unittest.main()
