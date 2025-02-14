(module
  (import "env" "memory" (memory 1))
  (func (export "E_3_3")
    (param $data0 i32)
    (param $data1 i32)
    (param $data2 i32)
    (param $data3 i32)
    (local $ridx0 i32)
    (local $ridx1 i32)
    (local $alu0 i32)
    (local $alu1 i32)
    (local $val0 i32)
    (local $val1 i32)
    (local $val2 i32)
    (local $alu2 i32)
    (local $alu3 i32)
    (local $alu4 i32)
    (local $alu5 i32)
    (local $alu6 i32)
    (local $alu7 i32)
    (local $alu8 i32)
    (local $alu9 i32)
    (local $alu10 i32)
    (local $alu11 i32)
    (local $alu12 i32)
    (local $alu13 i32)
    (local $alu14 i32)
    (local $alu15 i32)
    (local $alu16 i32)
    (local $alu17 i32)
    (local $alu18 i32)
    (local $alu19 i32)
    (local $alu20 i32)
    (local $alu21 i32)
    (local $alu22 i32)
    (local $alu23 i32)
    (local $alu24 i32)
    (local $alu25 i32)
    (local $alu26 i32)
    (local $alu27 i32)
    (local $alu28 i32)
    (local $alu29 i32)
    (local $alu30 i32)
    (local $alu31 i32)
    (local $alu32 i32)
    (local $alu33 i32)
    (local $alu34 i32)
    (local $alu35 i32)
    (local $alu36 i32)
    (local $alu37 i32)
    (local $alu38 i32)
    (local $alu39 i32)
    (local $alu40 i32)
    (local $alu41 i32)
    (local $alu42 i32)
    (local $alu43 i32)
    (local $alu44 i32)
    (local $alu45 i32)
    (local.set $ridx0 (i32.const 0))
    (block $block0
      (loop $loop0
        (br_if $block0
          (i32.eq
            (local.get $ridx0)
            (i32.const 3)
          )
        )
        (local.set $ridx1 (i32.const 0))
        (block $block1
          (loop $loop1
            (br_if $block1
              (i32.eq
                (local.get $ridx1)
                (i32.const 3)
              )
            )
            (local.set $alu0
              (i32.add
                (i32.mul
                  (local.get $ridx0)
                  (i32.const 3)
                )
                (local.get $ridx1)
              )
            )
            (local.set $alu1
              (i32.lt_s
                (local.get $alu0)
                (i32.const 5)
              )
            )
            (local.set $val0
              (i32.load
                (i32.add
                  (i32.mul
                    (i32.const 0)
                    (i32.const 4)
                  )
                  (local.get $data2)
                )
              )
            )
            (local.set $val1
              (i32.load
                (i32.add
                  (i32.mul
                    (i32.const 1)
                    (i32.const 4)
                  )
                  (local.get $data2)
                )
              )
            )
            (local.set $val2
              (i32.load
                (i32.add
                  (i32.mul
                    (local.get $alu0)
                    (i32.const 4)
                  )
                  (local.get $data1)
                )
              )
            )
            (local.set $alu2
              (select
                (local.get $val1)
                (i32.const 0)
                (local.get $alu1)
              )
            )
            (local.set $alu3
              (i32.xor
                (i32.xor
                  (local.get $val0)
                  (local.get $alu2)
                )
                (i32.const 466688986)
              )
            )
            (local.set $alu4
              (select
                (i32.add
                  (local.get $val1)
                  (i32.add
                    (local.get $val2)
                    (select
                      (i32.const 5)
                      (i32.const 0)
                      (local.get $alu1)
                    )
                  )
                )
                (i32.const 0)
                (local.get $alu1)
              )
            )
            (local.set $alu5
              (i32.add
                (i32.add
                  (local.get $val2)
                  (local.get $val0)
                )
                (local.get $alu4)
              )
            )
            (local.set $alu6
              (i32.xor
                (local.get $alu5)
                (i32.add
                  (i32.mul
                    (local.get $alu4)
                    (i32.const 8192)
                  )
                  (i32.div_u
                    (local.get $alu4)
                    (i32.const 524288)
                  )
                )
              )
            )
            (local.set $alu7
              (i32.add
                (local.get $alu5)
                (local.get $alu6)
              )
            )
            (local.set $alu8
              (i32.xor
                (local.get $alu7)
                (i32.add
                  (i32.mul
                    (local.get $alu6)
                    (i32.const 32768)
                  )
                  (i32.div_u
                    (local.get $alu6)
                    (i32.const 131072)
                  )
                )
              )
            )
            (local.set $alu9
              (i32.add
                (local.get $alu7)
                (local.get $alu8)
              )
            )
            (local.set $alu10
              (i32.xor
                (local.get $alu9)
                (i32.add
                  (i32.mul
                    (local.get $alu8)
                    (i32.const 67108864)
                  )
                  (i32.div_u
                    (local.get $alu8)
                    (i32.const 64)
                  )
                )
              )
            )
            (local.set $alu11
              (i32.add
                (local.get $alu9)
                (local.get $alu10)
              )
            )
            (local.set $alu12
              (i32.xor
                (local.get $alu11)
                (i32.add
                  (i32.mul
                    (local.get $alu10)
                    (i32.const 64)
                  )
                  (i32.div_u
                    (local.get $alu10)
                    (i32.const 67108864)
                  )
                )
              )
            )
            (local.set $alu13
              (i32.add
                (local.get $alu12)
                (local.get $alu3)
              )
            )
            (local.set $alu14
              (i32.add
                (i32.add
                  (local.get $alu11)
                  (local.get $alu2)
                )
                (local.get $alu13)
              )
            )
            (local.set $alu15
              (i32.xor
                (i32.add
                  (local.get $alu14)
                  (i32.const 1)
                )
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.mul
                        (local.get $alu12)
                        (i32.const 131072)
                      )
                      (i32.mul
                        (local.get $alu3)
                        (i32.const 131072)
                      )
                    )
                    (i32.div_u
                      (i32.add
                        (local.get $alu13)
                        (i32.const 1)
                      )
                      (i32.const 32768)
                    )
                  )
                  (i32.const 131072)
                )
              )
            )
            (local.set $alu16
              (i32.add
                (local.get $alu14)
                (local.get $alu15)
              )
            )
            (local.set $alu17
              (i32.xor
                (i32.add
                  (local.get $alu16)
                  (i32.const 1)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu15)
                    (i32.const 536870912)
                  )
                  (i32.div_u
                    (local.get $alu15)
                    (i32.const 8)
                  )
                )
              )
            )
            (local.set $alu18
              (i32.add
                (local.get $alu16)
                (local.get $alu17)
              )
            )
            (local.set $alu19
              (i32.xor
                (i32.add
                  (local.get $alu18)
                  (i32.const 1)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu17)
                    (i32.const 65536)
                  )
                  (i32.div_u
                    (local.get $alu17)
                    (i32.const 65536)
                  )
                )
              )
            )
            (local.set $alu20
              (i32.add
                (local.get $alu18)
                (local.get $alu19)
              )
            )
            (local.set $alu21
              (i32.xor
                (i32.add
                  (local.get $alu20)
                  (i32.const 1)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu19)
                    (i32.const 16777216)
                  )
                  (i32.div_u
                    (local.get $alu19)
                    (i32.const 256)
                  )
                )
              )
            )
            (local.set $alu22
              (i32.add
                (local.get $val0)
                (local.get $alu21)
              )
            )
            (local.set $alu23
              (i32.add
                (local.get $alu22)
                (i32.add
                  (local.get $alu20)
                  (local.get $alu3)
                )
              )
            )
            (local.set $alu24
              (i32.xor
                (i32.add
                  (local.get $alu23)
                  (i32.const 3)
                )
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.mul
                        (local.get $val0)
                        (i32.const 8192)
                      )
                      (i32.mul
                        (local.get $alu21)
                        (i32.const 8192)
                      )
                    )
                    (i32.div_u
                      (i32.add
                        (local.get $alu22)
                        (i32.const 2)
                      )
                      (i32.const 524288)
                    )
                  )
                  (i32.const 16384)
                )
              )
            )
            (local.set $alu25
              (i32.add
                (local.get $alu23)
                (local.get $alu24)
              )
            )
            (local.set $alu26
              (i32.xor
                (i32.add
                  (local.get $alu25)
                  (i32.const 3)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu24)
                    (i32.const 32768)
                  )
                  (i32.div_u
                    (local.get $alu24)
                    (i32.const 131072)
                  )
                )
              )
            )
            (local.set $alu27
              (i32.add
                (local.get $alu25)
                (local.get $alu26)
              )
            )
            (local.set $alu28
              (i32.xor
                (i32.add
                  (local.get $alu27)
                  (i32.const 3)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu26)
                    (i32.const 67108864)
                  )
                  (i32.div_u
                    (local.get $alu26)
                    (i32.const 64)
                  )
                )
              )
            )
            (local.set $alu29
              (i32.add
                (local.get $alu27)
                (local.get $alu28)
              )
            )
            (local.set $alu30
              (i32.xor
                (i32.add
                  (local.get $alu29)
                  (i32.const 3)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu28)
                    (i32.const 64)
                  )
                  (i32.div_u
                    (local.get $alu28)
                    (i32.const 67108864)
                  )
                )
              )
            )
            (local.set $alu31
              (i32.add
                (local.get $alu30)
                (local.get $alu2)
              )
            )
            (local.set $alu32
              (i32.add
                (i32.add
                  (local.get $val0)
                  (local.get $alu29)
                )
                (local.get $alu31)
              )
            )
            (local.set $alu33
              (i32.xor
                (i32.add
                  (local.get $alu32)
                  (i32.const 6)
                )
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.mul
                        (local.get $alu30)
                        (i32.const 131072)
                      )
                      (i32.mul
                        (local.get $alu2)
                        (i32.const 131072)
                      )
                    )
                    (i32.div_u
                      (i32.add
                        (local.get $alu31)
                        (i32.const 3)
                      )
                      (i32.const 32768)
                    )
                  )
                  (i32.const 393216)
                )
              )
            )
            (local.set $alu34
              (i32.add
                (local.get $alu32)
                (local.get $alu33)
              )
            )
            (local.set $alu35
              (i32.xor
                (i32.add
                  (local.get $alu34)
                  (i32.const 6)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu33)
                    (i32.const 536870912)
                  )
                  (i32.div_u
                    (local.get $alu33)
                    (i32.const 8)
                  )
                )
              )
            )
            (local.set $alu36
              (i32.add
                (local.get $alu34)
                (local.get $alu35)
              )
            )
            (local.set $alu37
              (i32.xor
                (i32.add
                  (local.get $alu36)
                  (i32.const 6)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu35)
                    (i32.const 65536)
                  )
                  (i32.div_u
                    (local.get $alu35)
                    (i32.const 65536)
                  )
                )
              )
            )
            (local.set $alu38
              (i32.add
                (local.get $alu36)
                (local.get $alu37)
              )
            )
            (local.set $alu39
              (i32.xor
                (i32.add
                  (local.get $alu38)
                  (i32.const 6)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu37)
                    (i32.const 16777216)
                  )
                  (i32.div_u
                    (local.get $alu37)
                    (i32.const 256)
                  )
                )
              )
            )
            (local.set $alu40
              (i32.add
                (local.get $alu39)
                (local.get $alu3)
              )
            )
            (local.set $alu41
              (i32.add
                (i32.add
                  (local.get $alu38)
                  (local.get $alu2)
                )
                (local.get $alu40)
              )
            )
            (local.set $alu42
              (i32.xor
                (i32.add
                  (local.get $alu41)
                  (i32.const 10)
                )
                (i32.add
                  (i32.add
                    (i32.add
                      (i32.mul
                        (local.get $alu39)
                        (i32.const 8192)
                      )
                      (i32.mul
                        (local.get $alu3)
                        (i32.const 8192)
                      )
                    )
                    (i32.div_u
                      (i32.add
                        (local.get $alu40)
                        (i32.const 4)
                      )
                      (i32.const 524288)
                    )
                  )
                  (i32.const 32768)
                )
              )
            )
            (local.set $alu43
              (i32.add
                (local.get $alu41)
                (local.get $alu42)
              )
            )
            (local.set $alu44
              (i32.xor
                (i32.add
                  (local.get $alu43)
                  (i32.const 10)
                )
                (i32.add
                  (i32.mul
                    (local.get $alu42)
                    (i32.const 32768)
                  )
                  (i32.div_u
                    (local.get $alu42)
                    (i32.const 131072)
                  )
                )
              )
            )
            (local.set $alu45
              (i32.add
                (local.get $alu43)
                (local.get $alu44)
              )
            )
            (f32.store
              (i32.add (i32.mul (local.get $alu0) (i32.const 4) ) (local.get $data0) )
              (f32.add
                (f32.reinterpret_i32
                  (i32.or
                    (i32.reinterpret_f32
                      (f32.const 1)
                    )
                    (i32.div_u
                      (i32.add
                        (i32.load
                          (i32.add
                            (i32.mul
                              (i32.add
                                (local.get $alu0)
                                (i32.const -5)
                              )
                              (i32.const 4)
                            )
                            (local.get $data3)
                          )
                        )
                        (select
                          (i32.add
                            (i32.add
                              (i32.add
                                (local.get $alu45)
                                (i32.xor
                                  (i32.add
                                    (local.get $alu45)
                                    (i32.const 10)
                                  )
                                  (i32.add
                                    (i32.mul
                                      (local.get $alu44)
                                      (i32.const 67108864)
                                    )
                                    (i32.div_u
                                      (local.get $alu44)
                                      (i32.const 64)
                                    )
                                  )
                                )
                              )
                              (local.get $alu3)
                            )
                            (i32.const 10)
                          )
                          (i32.const 0)
                          (local.get $alu1)
                        )
                      )
                      (i32.const 512)
                    )
                  )
                )
                (f32.const -1)
              )
            )
            (br $loop1
              (local.set $ridx1
                (i32.add
                  (local.get $ridx1)
                  (i32.const 1)
                )
              )
            )
          )
        )
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
