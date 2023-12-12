# 1.会通过指令调用进入global_dma
```C
void global_dma(int nodechip_idx, P_COMMAND p_command)
```

## 1.1 获取源数据数据和目标地址（目前不清楚这一段操作的功能）
```cpp
// 先取出记录的逻辑地址的高8bit，然后与剩下的低32bit组合在一起，形成完整的物理地址
u64 src_start_offset, dst_start_offset;
src_start_offset = (u64)desc.src_start_addr_h8;
dst_start_offset = (u64)desc.dst_start_addr_h8;
src_start_offset = (desc.src_start_addr_l32) + (src_start_offset << 32);
dst_start_offset = (desc.dst_start_addr_l32) + (dst_start_offset << 32);

// 这一步应该是进行调整
if (!desc.fill_constant_en) {
      src_start_offset = gdma_fix_ddr(nodechip_idx, src_start_offset);
    }
dst_start_offset = gdma_fix_ddr(nodechip_idx, dst_start_offset);


// 后续会判断目标起始地址是否在sram上
u32 is_dst_in_l2 = IN_L2_SRAM(dst_start_offset);
if (is_dst_in_l2 || IN_L2_SRAM(src_start_offset)) {
    mem_lock.lock();
}


// 再为数据分配一段连续地址
// convert addr ahead
CONTINUOUS_MEM *src_mem = NULL;
if (!need_fill) {
    src_mem = get_continuous_mem(nodechip_idx, src_start_offset);
    ASSERT_INFO(src_mem, "cannot get valid mem by src_addr = 0x%llx", src_start_offset);
    from_localmem = src_mem->type == MEM_TYPE_LOCAL;
    src_start_offset -= src_mem->start_addr;
}
CONTINUOUS_MEM *dst_mem = get_continuous_mem(nodechip_idx, dst_start_offset);
if (!dst_mem && desc.cmd_type == GDMA_TENSOR) {
  int dst_core_idx = ((dst_start_offset >> CORE_OFFSET_BIT) & (MAX_TPU_CORE_NUM - 1));
  dst_mem = get_continuous_mem(dst_core_idx, dst_start_offset);
}
```


## 1.2 核心操作dma_compress

```cpp
if (desc.cmd_type == GDMA_COMPRESS) {
  dma_compress(nodechip_idx, desc);
  MOVE_END;
}
```

 进入dma_compress后，首先会创建一个log_fp保存log信息,然后就是lmem相关的地址处理
 ```c
FILE *log_fp = nullptr;

// 然后也是一系列地址计算和地址分配
uint64_t src_laddr = ((uint64_t)desc.src_start_addr_h8 << 32) | desc.src_start_addr_l32;
uint64_t dst_gaddr = ((uint64_t)desc.dst_start_addr_h8 << 32) | desc.dst_start_addr_l32;
src_laddr = gdma_fix_ddr(nodechip_idx, src_laddr);
dst_gaddr = gdma_fix_ddr(nodechip_idx, dst_gaddr);
CONTINUOUS_MEM *src_lmem = get_continuous_mem(nodechip_idx, src_laddr);
CONTINUOUS_MEM *dst_gmem = get_continuous_mem(nodechip_idx, dst_gaddr);
ASSERT(src_lmem && dst_gmem && src_lmem->type == MEM_TYPE_LOCAL);
ASSERT((dst_gaddr % ALIGN_BYTES) == 0 && src_laddr % get_bytesize(desc.src_prec) == 0);
uint64_t src_loffset = src_laddr - src_lmem->start_addr;
uint64_t dst_goffset = dst_gaddr - dst_gmem->start_addr;

// 需要注意的是源数据大小小于LOCAL_MEM_SIZE时，才会开启racu
const int32_t start_npu = src_loffset / LOCAL_MEM_SIZE;

if (is_racu) {
    ASSERT_INFO(start_npu == 0, "racu compress start_npu must be 0, start_npu=%d",
                start_npu);
    ...

```

### 1.2.1 nnvlc v1.0的源码解析——nnvlc_encode
```cpp
int32_t in_size = src_shape.n * src_shape.c * src_shape.h
                      * src_shape.w * get_bytesize(dtype); // 如果dtype是8bit的，那么in_size=nchw;
                                                           // 如果是16bit，则in_size=2nchw
uint8_t *input = new uint8_t[in_size];   // 用于存储tensor数据
TENSOR_INFO tensor_info; 
tensor_info_generate(  // 创建一个结构体，保存n,c,h,w和stride信息
    src_shape.n,
    src_shape.c,
    src_shape.h,
    src_shape.w,
    src_loffset,
    3,
    &src_lstride,
    dtype,
    &tensor_info);
```
从lmem中数据读取到input中，其中读取的方式是按照c*h方向来读取
```cpp
gather_data_from_lmem(              
    get_local_mem(get_cur_nodechip_idx()),
    &tensor_info,
    input,
    false);
// 读取的函数为：
 for (int idx = 0; idx < (int)(n * c * h); idx++) {
    int nidx = idx / (c * h);
    int chidx = idx % (c * h);
    int cidx = chidx / h;
    int hidx = chidx % h;
    int n_offset = nidx * nstride;
    int c_offset = ((linfo.start_npu_idx + cidx) >> NPU_SHIFT) * cstride;
    int h_offset = hidx * hstride;
    const int npu_idx = (linfo.start_npu_idx + cidx) % NPU_NUM;
    T* npu_mem = (T*)(linfo.lmem->mem_arr[npu_idx]) + linfo.start_offset;
    f(npu_mem, idx, n_offset + c_offset + h_offset, cidx, hidx);
  }
```
可以发现是将w固定为1的方式取读取，其中f是lambda表达式，只要cidx和hidx没有都走到最后，就会按照设定方式复制数据到npu
```cpp
[&p_dst, &info](T* npu_mem, int idx, int offset, int cidx, int hidx) {
            if (cidx != (int)info->c - 1 || hidx < (int)info->matrix_col_margin) {
              memcpy(p_dst + idx, npu_mem + offset, sizeof(T));
            }
          }
```

进入核心函数nnvlc_encode
```
uint8_t *output =
    nnvlc_encode(input, in_size, dtype, desc.bias0, desc.bias1,
                 desc.is_signed, desc.zero_guard, out_size, log_fp,
                 dst_gaddr, src_loffset, src_shape, src_lstride);
```
完整代码:
- 创建一个encoder,会根据dtype(int8,fp16)选择对应encoder
- max_buf_s 是kmap_sz + blk_num * blk_len,blk_len（dtype是16bit则为32，是8bit则是16），blk_num=in_size / blk_len
- 创建uint8指针接收编码后的数据
- 进行编码encoder->encode
```cpp
uint8_t *nnvlc_encode(uint8_t *ibuf, int32_t isz, PREC dtype, uint8_t bias0,
                      uint8_t bias1, bool is_signed, bool zero_guard,
                      int32_t &osz, FILE *log_fp, gaddr_t saddr, laddr_t laddr,
                      shape_t shape, stride_t stride) {
  auto encoder = create_encoder(dtype, bias0, bias1, is_signed, zero_guard);
  int32_t max_buf_sz = encoder->max_enc_size(isz);
  max_buf_sz += sizeof(EncodeHeader);
  uint8_t *obuf = new (std::nothrow) uint8_t[max_buf_sz];
  memset(obuf, 0, max_buf_sz);
  assert(obuf);

  int32_t kmap_size = 0;
  int32_t enc_sz = encoder->encode(ibuf, isz, obuf + sizeof(EncodeHeader),
                                   log_fp, saddr + sizeof(EncodeHeader),
                                   laddr, shape, stride, kmap_size);
  delete encoder;

  // write header
  EncodeHeader header{};
  header.blk_enc_size = enc_sz;
  memcpy(obuf, &header, sizeof(header));

  osz = sizeof(header) + enc_sz + kmap_size;
  printf("size of header=%ld,encode_size=%d, kmap size=%d, compress datasize=%d\n",sizeof(header),enc_sz,kmap_size,osz);
  if (log_fp) {
    fprintf(log_fp, "encoder size:%d\n", enc_sz);
    fprintf(log_fp,
            "header: val:0x%08x_0x%08x_0x%08x_0x%08x, "
            "addr:0x%08lx\n",
            ((uint32_t *)obuf)[0], ((uint32_t *)obuf)[1], ((uint32_t *)obuf)[2],
            ((uint32_t *)obuf)[3], saddr);
  }
  return obuf;
}
```

**encoder->encode代码梳理**
```cpp
int32_t Float16VlcEncoder::encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf,
                                  FILE *log_fp, gaddr_t saddr, laddr_t laddr,
                                  shape_t shape, stride_t stride, int32_t &kmap_size) {
  assert((saddr & (NNVLC_ALIGN_BYTES - 1)) == 0);
  auto blk_num = calc_blk_num(isz);
  kmap_size = calc_kmap_sz(blk_num);
  // block encode
  BitStream kmap_strm(obuf, kmap_size, true); 
  BitStream payload_strm(obuf + kmap_size, blk_num << 5, true);

  CenterShift remapping;
  for (int32_t idx = 0, pos = 0; idx < blk_num; idx++, pos += 32) {
    uint8_t exp[16] = {0};
    uint8_t frac[16] = {0};
    uint8_t exp_buf[16] = {0};
    int32_t in_num = std::min(isz - pos, 32) >> 1; // 用于避免最后一个块小于32
    auto ptr = (uint16_t *)(ibuf + pos); # 将相连的两个8bit组合为16bit，一个32bit的块可以获得16个
    for (int32_t i = 0; i < 16; i++) {
      exp[i] = i < in_num ? (uint8_t)((ptr[i] >> 7) & 0xFF) : 0; // 取出高8部分,可能表示整数部分
      frac[i] = i < in_num ? (uint8_t)(((ptr[i] >> 15) << 7) | (ptr[i] & 0x7F)) : 0; // 低7Bit与最高位组成frac
      if (is_fp16 && zero_guard) {
        exp[i] = (exp[i] >> 3) == 0 ? 0 : exp[i];
      }
      exp_buf[i] = exp[i];
      exp[i] = remapping.transform(exp[i], bias0, zero_guard);
    }

    uint8_t znum = 0;
    int32_t pre_pos = payload_strm.pos();
    int32_t k = estimate_order_k(exp, zero_guard, log_fp);
    uint8_t ulen = block_encode(payload_strm, exp, k, zero_guard, znum);
    uint8_t k_info = (k == -1) ? 0xE0 : (k << 5) + ulen; // k=-1，默认224
    kmap_strm.write(&k_info, 8);
    // printf("%d \n",k_info);
    // write znum to kmap field
    if (zero_guard) {
      kmap_strm.write(&znum, 8);
    }

    for (int32_t i = 0; i < 16; i++) {
      if (exp[i] != 0 || !zero_guard) {
        payload_strm.write(&frac[i], 8);
      }
    }
  }
  int32_t blk_bs_size = align_up(((payload_strm.pos() + 7) >> 3), NNVLC_ALIGN_BYTES);
  return blk_bs_size;
```
- 先将obuf的前面部分放kmap，后面部分放数据，blk_num个块，且每个块32bit
- 将每个块组成16个uint16的值，然后取出exp和frac；
- frac直接写入payload；
- 其中exp会先transform，然后encode,最后写入kmap;并且exp会参与k的选择；
```
uint8_t CenterShift::transform(uint8_t val, uint8_t bias, bool zero_guard) {
  if (val == 0 && zero_guard)
    return 0;

  int16_t shift_data_i = val - bias;
  uint8_t range = (bias <= 128) ? bias : 255 - bias;
  if (bias <= 128) {
    // val >= 64,直接返回，反之有转无+zero_guard
    return (val >= (range << 1)) ? val
                                 : sign_to_unsign(shift_data_i) + zero_guard;
  } else {
    val < 2*bias - 255,则返回255-val+zero_guard;反之返回有转无+zero_guard
    return (val < (bias - range)) ? (range + bias - val + zero_guard)
                                  : (sign_to_unsign(shift_data_i) + zero_guard);
  }
}
```

```int32_t GREncoder::estimate_order_k(uint8_t *blk_in, bool zero_guard, FILE *log_fp) {
  int32_t best_k = 0;
  int32_t best_bs_size = 0x7FFFFFFF;
  // MAX_ORDER_K = 5
  for (int32_t k = 0; k <= (int)MAX_ORDER_K; k++) {
    uint8_t remain_field_size = k << 4; // 32个块，每个块kbit存储余数
    int32_t unary_field_len = 0;
    // ************ 基于k来计算商
    for (int32_t i = 0; i < 16; i++) {
      uint8_t quotient = blk_in[i] >> k;
      unary_field_len += (quotient + 1); // +1应该是留给0的
    }
    // ************
    // UNARY_FIELD_SIZE默认47？,由于总共多加16，所以实际上的商之和为32
    int32_t blk_size = (unary_field_len <= MAX_UNARY_FIELD_SIZE)
                           ? remain_field_size + unary_field_len
                           : 255;
    
    if (blk_size < best_bs_size) {
      best_k = k;
      best_bs_size = blk_size;
    }
    if (log_fp) {
      fprintf(log_fp, "  k:%d => map_len:%d\n", k, blk_size);
    }
  }
  best_k = (best_bs_size > 128) ? -1 : best_k;
  return best_k;
}
```
- 接着运行block_encode进行编码

![本地图片](/home/qcj/workcode/ultralytics/a.jpg)
```c++
 // bit plane encode for remain field
  for (int32_t k = 0; k < order_k; k++) {
    uint8_t bit_plane0 = 0, bit_plane1 = 0;

  // bit_val函数的功能是取出指定索引数据的第k位
  // static inline uint8_t bit_val(void *buf, int32_t byte_idx, int32_t bit_idx) {
  //   return (((uint8_t *)buf)[byte_idx] >> bit_idx) & 0x1;
  // }
    for (int32_t i = 0; i < 8; i++) {
      bit_plane0 |= (bit_val(blk_in, i, k) << i); // 低8bit用0表示
      bit_plane1 |= (bit_val(blk_in, i + 8, k) << i);  // 高位用1表示
    }
    remain_field[k << 1] = bit_plane0;
    remain_field[(k << 1) + 1] = bit_plane1;
  }
   // 将所有余数编码保存
  stream.write(remain_field, order_k << 4);

  /* if zero_guard is ture, the encoder
   * should add znum info to kmap field
   */
  if (zero_guard) {
    znum = 0;
    for (int32_t i = 0; i < 16; i++) {
      if (blk_in[i] == 0) {
        znum++;
      }
    }
  }
   
   // 对商进行编码
  // unary encode for unary field
  for (int32_t i = 0; i < 16; i++) {
    int32_t quotient = blk_in[i] >> order_k; // 商
    sym_end_pos_accum += (quotient + 1);  // pos+编码的长度
    sym_end_pos[i] = sym_end_pos_accum;
    int32_t byte_idx = sym_end_pos[i] / 8;
    int32_t bit_idx = sym_end_pos[i] % 8;
    unary_field[byte_idx] |= (1 << (bit_idx));
  }
  unary_field_len = sym_end_pos[15] + 1;

  assert(unary_field_len <= MAX_UNARY_FIELD_SIZE);

  uint8_t ulen = (unary_field_len - 16) & 0x1F;
  stream.write(unary_field, unary_field_len); // 保存
  return ulen; // 返回处0以外的1数量
```


### 1.2.1 nnvlc v2.0的源码解析——nnvlc2_encode
![](./racu.jpg)

只有is_racu为true才会进入v2.0编码,然后设置需要的strides,不是直接gather_data_from_lmem从local_mem中取数据；

```c++
if (is_racu) {
    ASSERT_INFO(start_npu == 0, "racu compress start_npu must be 0, start_npu=%d",
                start_npu);
    int type_len = get_bytesize(desc.src_prec);
    // hardware calculate racu offset using racu_stride * type_len, but racu is in byte unit
    // cmodel calculate racu offset without type_len, so here using stride * type_len as racu_stride
    // racu的stride，默认w=1
    stride_t racu_gstride = {
      .n = (int32_t)desc.src_wstride * type_len,
      .c = (int32_t)desc.dst_nstride * type_len,
      .h = (int32_t)desc.dst_cstride * type_len,
      .w = 1
    };
    stride_t meta_gstride = {
      .n = (int32_t)desc.dst_hstride,
      .c = (int32_t)desc.dst_wstride,
      .h = 1,
      .w = 1
    };
    uint64_t meta_gaddr = ((uint64_t)desc.mask_start_addr_h8 << 32) | desc.mask_start_addr_l32;
    meta_gaddr = gdma_fix_ddr(nodechip_idx, meta_gaddr);
    ASSERT(meta_gaddr % sizeof(meta_t) == 0);
    CONTINUOUS_MEM *meta_mem = get_continuous_mem(nodechip_idx, meta_gaddr);
    ASSERT(dst_gaddr + racu_gstride.n * src_shape.n <= dst_gmem->start_addr + dst_gmem->size);
    ASSERT(meta_mem && (meta_gaddr + meta_gstride.n * src_shape.n * sizeof(meta_t)
                        <= meta_mem->start_addr + meta_mem->size));
    nnvlc2_encode(dtype, src_shape, src_loffset, src_lstride, dst_gaddr,
                  racu_gstride, meta_gaddr, meta_gstride, desc.bias0,
                  desc.bias1, desc.is_signed, desc.zero_guard, log_fp);
  }
```

***进入核心部分nnvlc2_encode***

![](./c.jpg)

- C方向上切分，和NPU数量有关，lanec=（C+NPU_NUM-1）/NPU_NUM，也就意味着先从LMEM中收集NPU_NUM个数据，每个数据有w行，再对这NPU_NUM*w个数据进行压缩;
- 压缩后放置在GMEM上的shape将变为(n, lanec, h, gcw),其中gcw就是压缩后的最大可能的字节数，原大小是w*sizeof(dtype)

![](./d.jpg)
```c++
int lane_c = div_up(shape.c, NPU_NUM);
  int usize = unit_size(dtype);
  int max_isz = shape.w * usize * NPU_NUM;
  uint8_t *ibuf = new uint8_t[max_isz];
  auto encoder = create_encoder(dtype, bias0, bias1, is_signed, zero_guard);
  int max_obuf_sz = encoder->max_enc_size(max_isz);

  uint8_t *obuf = new uint8_t[max_obuf_sz];
  meta_t *meta_buf = new meta_t[shape.h];
  uint8_t *racu_ptr = (uint8_t *)get_continuous_mem_ptr(get_cur_nodechip_idx(), daddr);
  uint8_t *meta_ptr = (uint8_t *)get_continuous_mem_ptr(get_cur_nodechip_idx(), meta_addr);

  for (int n = 0; n < shape.n; n++) {
    for (int lc = 0; lc < lane_c; lc++) {
      int lanes = std::min(shape.c - NPU_NUM * lc, NPU_NUM);
      // 计算在第n个NPU上的地址
      laddr_t laddr = usize * (n * lstride.n + lc * lstride.c) + saddr;
      gaddr_t gaddr = n * racu_stride.n + lc * racu_stride.c + daddr;
      gaddr_t maddr = sizeof(meta_t) * (n * meta_stride.n + lc * meta_stride.c) + meta_addr;
      gaddr_t racu_gaddr = gaddr;
      meta_t *p_meta = meta_buf;
      for (int h = 0; h < shape.h; h++) {
        // copy racu data from lanes
        int isz = gather_racu(dtype, laddr, lanes, shape.w, ibuf);
        // do compress here
        int32_t kmap_size = 0;
        shape_t cur_shape = {1, lanes, 1, shape.w};
        // encode的操作与1.0一致
        int32_t enc_sz = encoder->encode(ibuf, isz, obuf, log_fp, racu_gaddr,
                                         laddr, cur_shape, lstride, kmap_size);

        int32_t offset = gaddr - racu_gaddr;
        assert(enc_sz % NNVLC_ALIGN_BYTES == 0 && (enc_sz >> NNVLC_ALIGN_SHIFT) < (1 << 12) && enc_sz >= 0);
        assert(offset % NNVLC_ALIGN_BYTES == 0 && (offset >> NNVLC_ALIGN_SHIFT) < (1 << 20) && offset >= 0);
        // store racu's meta to meta buf
        // 保存存放时的offset和压缩后大小
        p_meta->offset = offset >> NNVLC_ALIGN_SHIFT;
        p_meta->enc_sz = enc_sz >> NNVLC_ALIGN_SHIFT;
        // store racu to tpu->gmem

        // 写会GMEM
        memcpy(racu_ptr + racu_gaddr - daddr, obuf, enc_sz + kmap_size);

        if (log_fp) {
          fprintf(log_fp, "meta => val: 0x%08x addr:0x%08lx\n\n",
                  *((uint32_t *)p_meta), maddr + h * sizeof(meta_t));
        }
        laddr += lstride.h * usize;
        gaddr += racu_stride.h;
        racu_gaddr += (enc_sz + kmap_size);
        p_meta += 1;
      }
      // store meta data to tpu->gmem
      memcpy(meta_ptr + maddr - meta_addr, meta_buf, shape.h * sizeof(meta_t));
    }
  }

  delete[] ibuf;
  delete[] obuf;
  delete[] meta_buf;
  delete encoder;
}
```
总结：
1.确实提高寻址的速度，支持一定的随机访问；
2.NUP的标准化压缩；
缺点：
1.依然有冗余数据；




##  