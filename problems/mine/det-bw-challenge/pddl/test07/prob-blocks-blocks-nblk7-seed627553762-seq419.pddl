(define (problem blocks-nblk7-seed627553762-seq419)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b7) (on b2 b3) (on b3 b5) (on b4 b1) (on b5 b4) (ontable b6) (on b7 b6) (clear b2))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (on b3 b2) (on b4 b7) (ontable b5) (on b6 b1) (on b7 b3) (clear b5) (clear b6))))