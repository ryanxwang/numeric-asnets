(define (problem blocks-nblk5-seed896255376-seq5)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 - block)
    (:init (handempty) (on b1 b4) (ontable b2) (ontable b3) (ontable b4) (on b5 b3) (clear b1) (clear b2) (clear b5))
    (:goal (and (handempty) (ontable b1) (on b2 b1) (on b3 b4) (on b4 b2) (on b5 b3) (clear b5))))