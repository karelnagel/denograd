(module
  (import "env" "memory" (memory 1))
  (func (export "E_6n3")
    (param $data0 i32)
    (param $data1 i32)
    (local $ridx0 i32)
    
    ;; RANGE
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 6)
          )
        )
        
        ;; STORE
        (f32.store
          (i32.add
            (i32.mul
              (local.get $ridx0)
              (i32.const 4)
            )
            (local.get $data0)
          )
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

(module
  (import "env" "memory" (memory 1))
  (func (export "E_6n4")
    (param $data0 i32)
    (param $data1 i32)
    (local $ridx0 i32)
    
    ;; RANGE
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 6)
          )
        )
        
        ;; STORE
        (i32.store
          (i32.add
            (i32.mul
              (local.get $ridx0)
              (i32.const 4)
            )
            (local.get $data0)
          )
          (i32.load
            (i32.add
              (i32.mul
                (local.get $ridx0)
                (i32.const 4)
              )
              (local.get $data1)
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

(module
  (import "env" "memory" (memory 1))
  (func (export "E_6n5")
    (param $data0 i32)
    (param $data1 i32)
    (param $data2 i32)
    (local $ridx0 i32)
    
    ;; RANGE
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 6)
          )
        )
        
        ;; STORE
        (f32.store
          (i32.add
            (i32.mul
              (local.get $ridx0)
              (i32.const 4)
            )
            (local.get $data0)
          )
          (f32.add
            (f32.trunc_i32_s
              (i32.load
                (i32.add
                  (i32.mul
                    (local.get $ridx0)
                    (i32.const 4)
                  )
                  (local.get $data2)
                )
              )
            )
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