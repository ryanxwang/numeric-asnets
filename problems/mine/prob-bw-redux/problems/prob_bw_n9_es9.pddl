(define (problem prob_bw_9_n9_es9_r909)
  (:domain prob_bw)
  (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 - block)
  (:init (emptyhand) (on b1 b4) (on b2 b6) (on-table b3) (on-table b4) (on b5 b3) (on-table b6) (on b7 b8) (on-table b8) (on b9 b5) (clear b1) (clear b2) (clear b7) (clear b9))
  (:goal (and (emptyhand) (on-table b1) (on-table b2) (on-table b3) (on b4 b7) (on-table b5) (on b6 b8) (on b7 b6) (on-table b8) (on b9 b1) (clear b2) (clear b3) (clear b4) (clear b5) (clear b9)))
)