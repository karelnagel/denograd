(module
  (import "env" "memory" (memory 1))
  (func (export "E_4n5")
    (param $data0 i32)
    (param $data1 i32)
    
    ;; STORE
    (i32.store
      (i32.add (i32.mul (i32.const 0) (i32.const 1) ) (local.get $data0) )
      (i32.ne
        (i32.ne
          (f32.convert_i32_s
            (i32.load
              (i32.add
                (i32.mul
                  (i32.const 0)
                  (i32.const 4)
                )
                (local.get $data1)
              )
            )
          )
          (f32.const inf)
        )
        (i32.const 1)
      )
    )
    
    ;; STORE
    (i32.store
      (i32.add (i32.mul (i32.const 1) (i32.const 1) ) (local.get $data0) )
      (i32.ne
        (i32.ne
          (f32.convert_i32_s
            (i32.load
              (i32.add
                (i32.mul
                  (i32.const 1)
                  (i32.const 4)
                )
                (local.get $data1)
              )
            )
          )
          (f32.const inf)
        )
        (i32.const 1)
      )
    )
    
    ;; STORE
    (i32.store
      (i32.add (i32.mul (i32.const 2) (i32.const 1) ) (local.get $data0) )
      (i32.ne
        (i32.ne
          (f32.convert_i32_s
            (i32.load
              (i32.add
                (i32.mul
                  (i32.const 2)
                  (i32.const 4)
                )
                (local.get $data1)
              )
            )
          )
          (f32.const inf)
        )
        (i32.const 1)
      )
    )
    
    ;; STORE
    (i32.store
      (i32.add (i32.mul (i32.const 3) (i32.const 1) ) (local.get $data0) )
      (i32.ne
        (i32.ne
          (f32.convert_i32_s
            (i32.load
              (i32.add
                (i32.mul
                  (i32.const 3)
                  (i32.const 4)
                )
                (local.get $data1)
              )
            )
          )
          (f32.const inf)
        )
        (i32.const 1)
      )
    )
  )
)
