import collections
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContentTests(unittest.TestCase):
    def test_original_answers_preserved(self):
        text = (ROOT.parent / 'VLA具身智能面试题参考答案.md').read_text()
        actual = {s['id']: s for s in json.loads((ROOT / 'practice/data/sources.json').read_text())}
        for match in re.finditer(r'^## Q(\d+)\. ([^\n]+)\n', text, re.M):
            remaining = text[match.end():]
            stop = re.search(r'^#{1,2} ', remaining, re.M)
            body = remaining[:stop.start()] if stop else remaining
            item = actual[f'Q{match.group(1)}']
            self.assertEqual(item['answer'], body.strip())
            self.assertEqual(item['title'], match.group(2))

    def test_curated_bank(self):
        sources = json.loads((ROOT / 'practice/data/sources.json').read_text())
        questions = json.loads((ROOT / 'practice/data/questions.json').read_text())
        expected = {f'Q{i}' for i in [*range(1, 82), 102, 103, 104]}
        self.assertEqual({s['id'] for s in sources}, expected)
        self.assertEqual(len(sources), 84)
        source_map = {s['id']: s for s in sources}
        self.assertEqual(len(questions), 168)
        self.assertEqual(len({q['id'] for q in questions}), 168)
        coverage = collections.Counter((q['source_id'], q['type']) for q in questions)
        positions = collections.Counter()
        for s in sources:
            self.assertTrue(s['title'] and s['answer'] and s['chapter_title'])
            self.assertIsInstance(s['images'], list)
            self.assertIsInstance(s['citations'], list)
            self.assertNotIn('# 面试前请补齐', s['answer'])
            self.assertNotIn('# 第', s['answer'])
        for q in questions:
            with self.subTest(question=q['id']):
                source = source_map[q['source_id']]
                self.assertEqual(q['chapter'], source['chapter'])
                self.assertTrue(q['reviewed'])
                self.assertEqual(q['version'], 1)
                self.assertTrue(q['stem'] and q['explanation'] and q['evidence'])
                self.assertIn(q['evidence'], source['answer'])
                if q['type'] == 'single':
                    self.assertIn(len(q['options']), (2, 3, 4))
                    ids = {o['id'] for o in q['options']}
                    self.assertEqual(len(ids), len(q['options']))
                    self.assertEqual(len({o['text'] for o in q['options']}), len(ids))
                    self.assertEqual(len(q['correct']), 1)
                    self.assertIn(q['correct'][0], ids)
                    self.assertEqual(q['answers'], [])
                    positions[q['correct'][0]] += 1
                else:
                    self.assertEqual(q['type'], 'cloze')
                    self.assertEqual(q['options'], [])
                    self.assertEqual(q['correct'], [])
                    self.assertEqual(q['stem'].count('____'), len(q['answers']))
                    for aliases in q['answers']:
                        self.assertTrue(aliases)
                        self.assertTrue(all(isinstance(a, str) and a.strip() for a in aliases))
                        self.assertEqual(len(aliases), len(set(aliases)))
                    # Personal project details marked unverified must not be invented.
                    self.assertNotRegex(q['stem'], r'你实际(?:使用|采集|训练)了.*____')
        for sid in expected:
            self.assertEqual(coverage[sid, 'single'], 1)
            self.assertEqual(coverage[sid, 'cloze'], 1)
        self.assertEqual(set(positions), set('abcd'))
        self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)


if __name__ == '__main__':
    unittest.main()
