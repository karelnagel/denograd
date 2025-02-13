(module
  (import "env" "memory" (memory 1))
  (func (export "E_3")
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
            (i32.const 3)
          )
        )
        
        ;; STORE
        (i32.store
          (i32.add (i32.mul (local.get $ridx0) (i32.const 1) ) (local.get $data0) )
          (i32.load
            (i32.add
              (i32.mul
                (local.get $ridx0)
                (i32.const 1)
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
  (func (export "E_3n1")
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
            (i32.const 3)
          )
        )
        
        ;; STORE
        (i32.store
          (i32.add (i32.mul (local.get $ridx0) (i32.const 1) ) (local.get $data0) )
          (i32.ne
            (i32.load
              (i32.add
                (i32.mul
                  (local.get $ridx0)
                  (i32.const 1)
                )
                (local.get $data1)
              )
            )
            (i32.const 1)
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
