(module
  (import "env" "memory" (memory 1))
  (func (export "r_8_8")
    (param $data0 i32)
    (param $data1 i32)
    (local $ridx0 i32)
    
    ;; RANGE
    (local.set $ridx0 (i32.const 0))
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 8)
          )
        )
        
        ;; STORE
        (i32.store
          (i32.add (i32.mul (local.get $ridx0) (i32.const 4) ) (local.get $data0) )
          (i32.add
            (i32.load
              (i32.add
                (i32.mul
                  (i32.const 0)
                  (i32.const 4)
                )
                (local.get $data1)
              )
            )
            (i32.add
              (i32.add
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.add
                        (select
                          (i32.const 1)
                          (i32.const 0)
                          (i32.ne
                            (i32.lt_s
                              (local.get $ridx0)
                              (i32.const 6)
                            )
                            (i32.const 1)
                          )
                        )
                        (select
                          (i32.const 1)
                          (i32.const 0)
                          (i32.ne
                            (i32.lt_s
                              (local.get $ridx0)
                              (i32.const 7)
                            )
                            (i32.const 1)
                          )
                        )
                      )
                      (select
                        (i32.const 1)
                        (i32.const 0)
                        (i32.ne
                          (i32.lt_s
                            (local.get $ridx0)
                            (i32.const 5)
                          )
                          (i32.const 1)
                        )
                      )
                    )
                    (select
                      (i32.const 1)
                      (i32.const 0)
                      (i32.ne
                        (i32.lt_s
                          (local.get $ridx0)
                          (i32.const 4)
                        )
                        (i32.const 1)
                      )
                    )
                  )
                  (select
                    (i32.const 1)
                    (i32.const 0)
                    (i32.ne
                      (i32.lt_s
                        (local.get $ridx0)
                        (i32.const 3)
                      )
                      (i32.const 1)
                    )
                  )
                )
                (select
                  (i32.const 1)
                  (i32.const 0)
                  (i32.ne
                    (i32.lt_s
                      (local.get $ridx0)
                      (i32.const 2)
                    )
                    (i32.const 1)
                  )
                )
              )
              (select
                (i32.const 1)
                (i32.const 0)
                (i32.ne
                  (i32.lt_s
                    (local.get $ridx0)
                    (i32.const 1)
                  )
                  (i32.const 1)
                )
              )
            )
          )
        )
        
        ;; ENDRANGE
        (br $loop0
          (local.set $ridx0
            (i32.add
              (local.get $ridx0)
              (i32.const 1)
            )
          )
        )
      )
    )
  )
)