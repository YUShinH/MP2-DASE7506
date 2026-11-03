"""Check interaction behavior and custom derivatives independently of fitted scores."""
import unittest
import numpy as np
from selector import linear_input,bilinear_logits
from search import check_gradients


class QuestionInteractionTests(unittest.TestCase):
    def test_additive_question_cancels_between_candidates(self):
        base=np.zeros((1,4,2),dtype=np.float32); r=np.array([[[1.,0.],[0.,1.],[1.,1.],[0.,0.]]]); q=np.array([[1.,2.]])
        a=linear_input(base,q,r,False); b=linear_input(base,q+3,r,False)
        np.testing.assert_allclose(a[:,0]-a[:,1],b[:,0]-b[:,1])
    def test_interactions_can_change_preference(self):
        base=np.zeros((1,4,2),dtype=np.float32); r=np.array([[[1.,0.],[0.,1.],[1.,1.],[0.,0.]]])
        a=linear_input(base,np.array([[1.,0.]]),r,True); b=linear_input(base,np.array([[0.,1.]]),r,True)
        # The first product coordinate occurs after base, solution and question.
        self.assertGreater((a[:,0]-a[:,1])[0,6],0); self.assertEqual((b[:,0]-b[:,1])[0,6],0)
    def test_bilinear_question_changes_score(self):
        x=np.array([[[1.,0.],[0.,1.],[1.,1.],[0.,0.]]]); w=dict(coef=np.zeros(2),bias=np.zeros(1),u=np.eye(2),v=np.eye(2))
        a=bilinear_logits(x,np.array([[1.,0.]]),w); b=bilinear_logits(x,np.array([[0.,1.]]),w)
        self.assertGreater(a[0,0],a[0,1]); self.assertGreater(b[0,1],b[0,0])
    def test_analytic_derivatives(self):
        self.assertLess(check_gradients(),1e-6)


if __name__=='__main__': unittest.main()
