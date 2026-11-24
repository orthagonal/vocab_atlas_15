"""Run: python -m unittest -v test_atlas.py"""
import json
import tempfile
import unittest
from pathlib import Path

from vocab_atlas_15 import surface_bytes, detect_encoding, read_vocabulary, infer_model_type, load_vocab, analyze


class VocabAtlasTests(unittest.TestCase):
    def test_bytelevel_decoding(self):
        self.assertEqual(surface_bytes('Ġcat', 'bytelevel'), b' cat')
        self.assertEqual(surface_bytes('Ċ', 'bytelevel'), b'\n')
        self.assertEqual(surface_bytes('Ã©', 'bytelevel'), 'é'.encode('utf-8'))
        self.assertEqual(surface_bytes('Â', 'bytelevel'), b'\xc2')  # incomplete byte sequence preserved

    def test_unigram_and_specials(self):
        data = {'model':{'type':'Unigram','vocab':[['▁cat',-1.0],['dog',-2.0],['<unk>',0]]},
                'decoder':{'type':'Metaspace'},
                'added_tokens':[{'id':2,'content':'<unk>','special':True},
                                {'id':3,'content':'other','special':False}]}
        vocab, special = read_vocabulary(data)
        self.assertEqual(len(vocab), 4)
        self.assertEqual(special, {'<unk>'})
        self.assertEqual(surface_bytes('▁cat', detect_encoding(data)), b' cat')

    def test_wordpiece(self):
        data={'model':{'type':'WordPiece','vocab':{'cat':0,'##cat':1},'continuing_subword_prefix':'##'}}
        vocab, special=read_vocabulary(data)
        self.assertEqual(set(vocab)-special, {'cat','##cat'})
        self.assertEqual(surface_bytes('##cat','wordpiece'), b'cat')

    def test_gpt2_like_json_without_model_type(self):
        # Some Hugging Face tokenizer.json exports omit model.type entirely.
        # This previously raised KeyError('type') in load_vocab.
        data = {
            'pre_tokenizer': {'type':'ByteLevel','add_prefix_space':False},
            'decoder': {'type':'ByteLevel'},
            'model': {
                'vocab': {'a':0,'Ġthe':1,'Ċ':2},
                'merges': ['a b', 'Ġ t'],
            },
            'added_tokens': [{'id':3, 'content':'<|endoftext|>', 'special':True}],
        }
        self.assertEqual(infer_model_type(data), 'BPE (inferred)')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            path=folder/'openai-community__gpt2.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            v=load_vocab('GPT-2', 'openai-community/gpt2', folder)
            self.assertEqual(v.model_type, 'BPE (inferred)')
            self.assertEqual(v.encoding, 'bytelevel')
            self.assertEqual(v.base_size, 3)
            self.assertEqual(v.full_size, 4)
            self.assertEqual(v.special_size, 1)
            self.assertEqual(v.raw, {'a','Ġthe','Ċ'})
            self.assertIn(b' the', v.surface)

    def test_inference_preserves_declared_model_type(self):
        self.assertEqual(infer_model_type({'model': {'type':'BPE', 'vocab':{}}}), 'BPE')
        self.assertEqual(infer_model_type({'model': {'vocab':[['▁x',-1.0]]}}), 'Unigram (inferred)')
        self.assertEqual(infer_model_type({'model': {'vocab':{'##x':1}, 'continuing_subword_prefix':'##'}}), 'WordPiece (inferred)')
        self.assertEqual(infer_model_type({'model': {'vocab':{'x':1}}}), 'Unspecified mapping vocabulary')

    def test_pipeline_with_fixture_repos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'input'
            inputs.mkdir()
            examples=[
                ('byte', 'fake/byte', {'model':{'type':'BPE','vocab':{'Ġcat':0,'dog':1,'no':2}},
                        'decoder':{'type':'ByteLevel'}}),
                ('sentencepiece','fake/uni', {'model':{'type':'Unigram','vocab':[['▁cat',-1],['cat',-2],['dog',-3]]},
                        'decoder':{'type':'Metaspace'}}),
                ('wordpiece','fake/wp', {'model':{'type':'WordPiece','vocab':{'cat':0,'##cat':1,'tree':2}}}),
            ]
            vocabs=[]
            for name,repo,data in examples:
                (inputs/(repo.replace('/','__')+'.json')).write_text(json.dumps(data),encoding='utf-8')
                vocabs.append(load_vocab(name,repo,inputs))
            summary=analyze(vocabs,root/'result',make_plots=False,write_full_index=True)
            self.assertEqual(summary['model_count'],3)
            self.assertEqual(summary['raw_token_union'],7)
            self.assertEqual(summary['normalized_byte_surface_union'],5)
            self.assertEqual(summary['raw_tokens_found_in_every_vocabulary'],0)
            self.assertEqual(summary['raw_tokens_found_in_exactly_one_vocabulary'],5)
            self.assertTrue((root/'result'/'REPORT.md').exists())
            self.assertTrue((root/'result'/'all_token_prevalence.csv.gz').exists())


if __name__ == '__main__':
    unittest.main()
