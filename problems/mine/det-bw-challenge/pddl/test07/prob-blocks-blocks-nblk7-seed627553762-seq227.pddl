(define (problem blocks-nblk7-seed627553762-seq227)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (ontable b1) (on b2 b7) (on b3 b6) (on b4 b2) (on b5 b4) (ontable b6) (on b7 b1) (clear b3) (clear b5))
    (:goal (and (handempty) (on b1 b7) (on b2 b1) (on b3 b6) (on b4 b3) (on b5 b4) (ontable b6) (ontable b7) (clear b2) (clear b5))))