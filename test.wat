(module
  (import "env" "memory" (memory 1))
  (func (export "E_7n1")
    (param $data0 i32)
    (param $data1 i32)
    (local $ridx0 i32)
    ;; Ops.RANGE
    (local.set $ridx0 (i32.const 0))
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 7)
          )
        )
        ;; Ops.STORE
        (i32.store8
          (i32.add
            (i32.mul
              (local.get $ridx0)
              (i32.const 1)
            )
            (local.get $data0)
          )
          (i32.ne
            (i32.trunc_f32_s
              (f32.load
                (i32.add
                  (i32.mul
                    (local.get $ridx0)
                    (i32.const 4)
                  )
                  (local.get $data1)
                )
              )
            )
            (i32.const 1)
          )
        )
        ;; Ops.ENDRANGE
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
