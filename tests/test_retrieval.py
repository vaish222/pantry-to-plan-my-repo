"""Offline retrieval and indexing regression tests; no live API calls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import index_recipes as indexing
from pipeline import generate_plan
from recipes import RECIPES
from retrieval import AutoRetriever, LocalRetriever, PineconeRetriever, create_retriever
from schemas import PantryItem, PantryState, PlanningRequest


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name, value in {
            'CACHE_DIR': root,
            'EMBEDDINGS_FILE': root / 'embeddings.json',
            'VECTORIZER_FILE': root / 'vectorizer.json',
        }.items():
            patcher = patch.object(indexing, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.request = PlanningRequest(cuisines=['indian'], dinner_calorie_target=600,
                                       days=3, vegetarian_required=True, goal='general')
        self.pantry = PantryState(items=[])

    def test_local_with_cloud_credentials_never_indexes_cloud(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'PINECONE_API_KEY': 'fake',
                                      'PINECONE_INDEX_NAME': 'fake', 'RETRIEVAL_BACKEND': 'local'}), \
             patch.object(indexing, 'index_recipes_to_pinecone', side_effect=AssertionError('cloud called')):
            retriever = create_retriever()
            candidates = retriever.search(self.pantry, self.request, 40)
        expected = {r.recipe_id for r in RECIPES if r.vegetarian}
        self.assertEqual({c.recipe_id for c in candidates}, expected)
        plan = generate_plan(self.pantry, self.request, retriever, temperature=0)
        self.assertEqual(len(plan.day_plans), 3)
        self.assertTrue(all(day.recipe.vegetarian for day in plan.day_plans))

    def test_expected_recipe_in_top_k_for_each_cuisine(self):
        retriever = LocalRetriever()
        for cuisine in ('italian', 'american', 'chinese', 'indian'):
            recipe = next(r for r in RECIPES if r.vegetarian and cuisine in r.cuisine_tags)
            pantry = PantryState(items=[PantryItem(ingredient_id=i.ingredient_id,
                display_name=i.ingredient_id, quantity_g=i.quantity_g, confidence=1)
                for i in recipe.ingredients])
            request = self.request.model_copy(update={'cuisines': [cuisine]})
            self.assertIn(recipe.recipe_id, [c.recipe_id for c in retriever.search(pantry, request, 5)])

    def test_cached_tfidf_matches_fresh_and_invalidates_changed_corpus(self):
        _, _, fresh = indexing.index_recipes()
        _, _, cached = indexing.index_recipes()
        query = ['brown_rice indian dinner']
        np.testing.assert_allclose(fresh.transform(query).toarray(), cached.transform(query).toarray())
        changed = [RECIPES[0].model_copy(update={'title': 'Different recipe text'}), *RECIPES[1:]]
        with patch.object(indexing, 'RECIPES', changed):
            self.assertIsNone(indexing.load_embeddings())

    def cloud_retriever(self, matches):
        retriever = PineconeRetriever.__new__(PineconeRetriever)
        retriever.namespace = 'recipes'
        retriever.recipes_by_id = {r.recipe_id: r for r in RECIPES}
        retriever.index = Mock()
        retriever.index.query.return_value = {'matches': matches}
        retriever._embed_text = Mock(return_value=[0.1, 0.2])
        return retriever

    def test_cloud_filter_and_defensive_local_gate(self):
        veg = next(r for r in RECIPES if r.vegetarian)
        meat = next(r for r in RECIPES if not r.vegetarian)
        retriever = self.cloud_retriever([{'id': meat.recipe_id, 'score': 1},
                                         {'id': veg.recipe_id, 'score': .8}])
        result = retriever.search(self.pantry, self.request, 5)
        self.assertEqual([c.recipe_id for c in result], [veg.recipe_id])
        self.assertEqual(retriever.index.query.call_args.kwargs['filter'], {'vegetarian': {'$eq': True}})
        request = self.request.model_copy(update={'vegetarian_required': False})
        self.assertEqual(len(retriever.search(self.pantry, request, 5)), 2)
        self.assertNotIn('filter', retriever.index.query.call_args.kwargs)

    def test_unknown_cloud_recipe_is_error(self):
        retriever = self.cloud_retriever([{'id': 'unknown_recipe', 'score': 1}])
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_RECIPE_ID: unknown_recipe'):
            retriever.search(self.pantry, self.request, 5)

    def test_embedding_cache_reuse_and_invalidation(self):
        client = Mock()
        client.embeddings.create.side_effect = lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25] * kw['dimensions'])])
        for _ in range(2):
            indexing.cached_cloud_embedding(client, 'rice', 'model-a', 2)
        self.assertEqual(client.embeddings.create.call_count, 1)
        for text, model, dims in [('beans', 'model-a', 2), ('rice', 'model-b', 2), ('rice', 'model-a', 3)]:
            indexing.cached_cloud_embedding(client, text, model, dims)
        self.assertEqual(client.embeddings.create.call_count, 4)

    def test_initialization_failure_can_fall_back_offline(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'PINECONE_API_KEY': 'fake',
                                      'PINECONE_INDEX_NAME': 'fake', 'RETRIEVAL_BACKEND': 'auto'}), \
             patch('retrieval.PineconeRetriever', side_effect=ConnectionError('offline')):
            retriever = create_retriever()
            self.assertTrue(retriever.search(self.pantry, self.request, 5))

    def test_auto_success_does_not_build_local(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'PINECONE_API_KEY': 'fake',
                                      'PINECONE_INDEX_NAME': 'fake'}, clear=True), \
             patch('retrieval.PineconeRetriever') as cloud, \
             patch('retrieval.LocalRetriever') as local:
            expected = [Mock(recipe_id='example')]
            cloud.return_value.search.return_value = expected
            retriever = create_retriever()
            self.assertIsInstance(retriever, AutoRetriever)
            self.assertEqual(retriever.search(self.pantry, self.request), expected)
            local.assert_not_called()
            self.assertEqual(retriever.backend, 'pinecone')

    def test_query_failure_switches_to_local_for_later_calls(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'PINECONE_API_KEY': 'fake',
                                      'PINECONE_INDEX_NAME': 'fake'}, clear=True), \
             patch('retrieval.PineconeRetriever') as cloud:
            cloud.return_value.search.side_effect = TimeoutError('offline')
            retriever = create_retriever()
            first = retriever.search(self.pantry, self.request, 5)
            self.assertTrue(first)
            self.assertEqual(first, retriever.search(self.pantry, self.request, 5))
            self.assertEqual(cloud.return_value.search.call_count, 1)
            self.assertEqual(retriever.backend, 'local')
            self.assertIsNotNone(retriever.fallback_reason)
            self.assertTrue(all(next(r for r in RECIPES if r.recipe_id == c.recipe_id).vegetarian
                                for c in first))

    def test_auto_missing_credentials_uses_local(self):
        with patch.dict('os.environ', {}, clear=True), \
             patch('retrieval.PineconeRetriever') as cloud:
            retriever = create_retriever()
            self.assertTrue(retriever.search(self.pantry, self.request, 5))
            cloud.assert_not_called()
            self.assertEqual(retriever.backend, 'local')

    def test_empty_results_and_unknown_ids_do_not_trigger_fallback(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake', 'PINECONE_API_KEY': 'fake',
                                      'PINECONE_INDEX_NAME': 'fake'}, clear=True), \
             patch('retrieval.PineconeRetriever') as cloud, \
             patch('retrieval.LocalRetriever') as local:
            retriever = create_retriever()
            cloud.return_value.search.return_value = []
            self.assertEqual(retriever.search(self.pantry, self.request), [])
            cloud.return_value.search.side_effect = ValueError('UNKNOWN_RECIPE_ID: bad')
            with self.assertRaisesRegex(ValueError, 'UNKNOWN_RECIPE_ID'):
                retriever.search(self.pantry, self.request)
            local.assert_not_called()

    def test_explicit_pinecone_reports_errors(self):
        with patch('retrieval.PineconeRetriever', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError):
                create_retriever('pinecone')


if __name__ == '__main__':
    unittest.main()
