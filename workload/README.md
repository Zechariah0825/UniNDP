### Workload File Format Description

!!! Note that there must be an empty line at the start and end of the file !!!

- The files in this directory are workload files for PNMulator, used to simulate NN
- The file format is csv, with each row representing a network layer, formatted as:
    - Name
    - Operator type
    - input size
    - shared dimension
    - weight size
    - batch size
- Example resnet18_32.csv is the cifar10 version of resnet18, with an input size of 32x32

### How to processing QKt and SV GEMM (head and context_length ) dimension to adopt "(M,K,L,B)" dimension setted by mm_micro dimension

- For QKt: 
  - Q：(1 x (nheadx(dmodel/nhead))), (M x K) -> (1 x dmodel)
  - K：((nhead x (dmodel/nhead)) x context_len), (K x L) -> (dmodel x context_len)
  - Concatting various q(head) and k(head) matrix across head dimension.

- For SV: 
  - S：(1 x(nhead x context_len)), (M x K) -> (1 x context_len)
  - V：((nhead x context_len) x (demodel/nhead)), (K x L) -> (nhead context_len x dhead)
  - Concatting various s(head) and v(head) matrix across context_len dimension.