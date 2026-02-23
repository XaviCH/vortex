# Code readability

## Functions
The methods to know the scope of a function is as follows:

- local_ is used to require that all threads in a local group are active.
- sub_group_  is used to require that all threads in a sub group are active.

otherwise, function is expected to not requiere any type of sync.

## Pointers

- g_ is used to reference a pointer to global memory.
- a_ is used to reference a value reference to global memory.
- l_ is used to reference a pointer to local memory.
- c_ is used to reference a pointer to constant memory.

otherwise, pointer is expected to be private for each thread.

