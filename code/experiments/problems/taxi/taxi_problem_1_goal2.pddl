
(define (problem taxiproblem)
(:domain taxi)
(:objects
    loc0 loc1 loc2 loc3 loc4 loc5 loc6 loc7 loc8 loc9 loc10 loc11 loc12 loc13 loc14 loc15 loc16 loc17 loc18 loc19 loc20 loc21 loc22 loc23 loc24 - loc
    north south east west - dir
)
(:init
(= (total-cost) 0)

(move-dir loc0 loc5 south)
(move-dir loc1 loc6 south)
(move-dir loc2 loc7 south)
(move-dir loc3 loc8 south)
(move-dir loc4 loc9 south)
(move-dir loc5 loc10 south)
(move-dir loc6 loc11 south)
(move-dir loc7 loc12 south)
(move-dir loc8 loc13 south)
(move-dir loc9 loc14 south)
(move-dir loc10 loc15 south)
(move-dir loc11 loc16 south)
(move-dir loc12 loc17 south)
(move-dir loc13 loc18 south)
(move-dir loc14 loc19 south)
(move-dir loc15 loc20 south)
(move-dir loc16 loc21 south)
(move-dir loc17 loc22 south)
(move-dir loc18 loc23 south)
(move-dir loc19 loc24 south)
(move-dir loc20 loc20 south)
(move-dir loc21 loc21 south)
(move-dir loc22 loc22 south)
(move-dir loc23 loc23 south)
(move-dir loc24 loc24 south)
(move-dir loc0 loc0 north)
(move-dir loc1 loc1 north)
(move-dir loc2 loc2 north)
(move-dir loc3 loc3 north)
(move-dir loc4 loc4 north)
(move-dir loc5 loc0 north)
(move-dir loc6 loc1 north)
(move-dir loc7 loc2 north)
(move-dir loc8 loc3 north)
(move-dir loc9 loc4 north)
(move-dir loc10 loc5 north)
(move-dir loc11 loc6 north)
(move-dir loc12 loc7 north)
(move-dir loc13 loc8 north)
(move-dir loc14 loc9 north)
(move-dir loc15 loc10 north)
(move-dir loc16 loc11 north)
(move-dir loc17 loc12 north)
(move-dir loc18 loc13 north)
(move-dir loc19 loc14 north)
(move-dir loc20 loc15 north)
(move-dir loc21 loc16 north)
(move-dir loc22 loc17 north)
(move-dir loc23 loc18 north)
(move-dir loc24 loc19 north)
(move-dir loc0 loc1 east)
(move-dir loc1 loc1 east)
(move-dir loc2 loc3 east)
(move-dir loc3 loc4 east)
(move-dir loc4 loc4 east)
(move-dir loc5 loc6 east)
(move-dir loc6 loc6 east)
(move-dir loc7 loc8 east)
(move-dir loc8 loc9 east)
(move-dir loc9 loc9 east)
(move-dir loc10 loc11 east)
(move-dir loc11 loc12 east)
(move-dir loc12 loc13 east)
(move-dir loc13 loc14 east)
(move-dir loc14 loc14 east)
(move-dir loc15 loc15 east)
(move-dir loc16 loc17 east)
(move-dir loc17 loc17 east)
(move-dir loc18 loc19 east)
(move-dir loc19 loc19 east)
(move-dir loc20 loc20 east)
(move-dir loc21 loc22 east)
(move-dir loc22 loc22 east)
(move-dir loc23 loc24 east)
(move-dir loc24 loc24 east)
(move-dir loc0 loc0 west)
(move-dir loc1 loc0 west)
(move-dir loc2 loc2 west)
(move-dir loc3 loc2 west)
(move-dir loc4 loc3 west)
(move-dir loc5 loc5 west)
(move-dir loc6 loc5 west)
(move-dir loc7 loc7 west)
(move-dir loc8 loc7 west)
(move-dir loc9 loc8 west)
(move-dir loc10 loc10 west)
(move-dir loc11 loc10 west)
(move-dir loc12 loc11 west)
(move-dir loc13 loc12 west)
(move-dir loc14 loc13 west)
(move-dir loc15 loc15 west)
(move-dir loc16 loc16 west)
(move-dir loc17 loc16 west)
(move-dir loc18 loc18 west)
(move-dir loc19 loc18 west)
(move-dir loc20 loc20 west)
(move-dir loc21 loc21 west)
(move-dir loc22 loc21 west)
(move-dir loc23 loc23 west)
(move-dir loc24 loc23 west)

(pasloc-at-loc red loc0)
(pasloc-at-loc green loc4)
(pasloc-at-loc yellow loc20)
(pasloc-at-loc blue loc23)
(taxi-at loc6)
(passenger-at green)
)
(:goal
(and
(passenger-at blue)
)
)
(:metric minimize (total-cost))
)
