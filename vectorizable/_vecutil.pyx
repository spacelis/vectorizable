def vec_dim_align(arrs):
    ''' Properly padding the nested arrays (tensors)'''
    maxlens = tuple()
    for e in arrs:
        if isinstance(e, (list, tuple)):
            lens = vec_dim_align(e)
            maxlens = tuple(
                max(a, b)
                for a, b in zip(lens, maxlens + ((0,) * (len(lens) - len(maxlens)))))
    return (len(arrs),) + maxlens
