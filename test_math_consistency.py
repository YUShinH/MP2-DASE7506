"""Boundary/false-positive tests for the arithmetic verifier."""
import unittest
from math_consistency import inspect,numeric


class ArithmeticChecks(unittest.TestCase):
    def test_supported_equations(self):
        for text in ['(2 + 3) * 4 = 20',r'\frac{10+15+5}{2}=\frac{30}{2}=15',r'\frac{1}{\frac{2}{3}}=1.5','25% of 80 = 20','1 1/2 + 1/2 = 2','-1 1/2 + 2 = 0.5','1,200 + 300 = 1,500',r'2^{3}=8','2(3+4)=14','(2+3)(4-1)=15','-3 + 1 = -2',r'\left(2+3\right)\times 4=20']:
            with self.subTest(text=text):
                r=inspect(text); self.assertGreater(r['checked'],0); self.assertEqual(r['counts']['wrong'],0)
    def test_wrong_chain(self):
        r=inspect('2+3=5=6'); self.assertEqual(r['counts']['exact'],1); self.assertEqual(r['counts']['wrong'],1)
    def test_rounding(self):
        self.assertEqual(inspect('1/3=0.33')['counts']['rounded'],1)
        self.assertEqual(inspect('1/3=0')['counts']['wrong'],1)
        self.assertEqual(inspect('1/3≈0')['counts']['rounded'],1)
        self.assertEqual(inspect('1/3=33.33%')['counts']['rounded'],1)
        self.assertEqual(inspect('(1/3)*100=33.33%')['counts']['rounded'],1)
    def test_undefined(self):
        self.assertEqual(inspect('1/0=0')['counts']['undefined'],1)
    def test_unsupported_not_wrong(self):
        for text in ['x+2=3','2*x+1=3','unknown = 3','sqrt(4)=2','2x=4','2+x=3','2+foo=3','2+3=6x']:
            with self.subTest(text=text): self.assertEqual(inspect(text)['counts']['wrong'],0)
    def test_final_result_is_signal_not_proof(self):
        r=inspect('2+3=5\n#### 7'); self.assertTrue(r['last_differs_answer']); self.assertEqual(r['counts']['wrong'],0)
        self.assertTrue(inspect(r'\boxed{3} then \boxed{4}')['conflicting_markers'])
    def test_safe_bounds(self):
        for expression in ["__import__('os').system('echo unsafe')",'2**999999','[1][0]','1e999','x+2']:
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError,SyntaxError)): numeric(expression)
    def test_unknown_is_reported(self):
        self.assertGreater(inspect('x+2=3')['counts']['unknown'],0)
        self.assertEqual(inspect('.5 + .5 = 1')['counts']['exact'],1)
    def test_real_text_boundaries(self):
        for text in [r'60\% \text{ of } 2000 = 0.6 \times 2000 = 1200','Half of 28 = **14 students**',r'\frac{1}{4} \text{ of } 400 = 100', '- $20 * 30 = $600',r'70+42=\boxed{112}','one third of 12 = 4','sum of 1 and 2 = 3','- 20 * 30 = 600','- Half of 28 = 14',r'\frac{8}{25} * 100 = 32%',r'\text{quantity} + 3 = 4 * 8 + 3']:
            with self.subTest(text=text): self.assertEqual(inspect(text)['counts']['wrong'],0)


if __name__=='__main__': unittest.main()
