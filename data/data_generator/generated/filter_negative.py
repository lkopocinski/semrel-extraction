
# positive_pairs = []
#
# with open('positive.context') as f_pos:
#     for line in f_pos:
#         line = line.strip()
#         by_tab = line.split('\t')
#
#         idx_from, ctx_from = by_tab[0].split(':', 1)
#         idx_to, ctx_to = by_tab[1].split(':', 1)
#
#         ctx_from = eval(ctx_from)
#         ctx_to = eval(ctx_to)
#
#         idx_from = int(idx_from)
#         idx_to = int(idx_to)
#
#         positive_pairs.append((ctx_from[idx_from], ctx_to[idx_to]))
#         positive_pairs.append((ctx_to[idx_to], ctx_from[idx_from]))

with open('negative.fixed.filtered.context') as f_pos:
    for line in f_pos:
        line = line.strip()
        by_tab = line.split('\t')

        idx_from, ctx_from = by_tab[0].split(':', 1)
        idx_to, ctx_to = by_tab[1].split(':', 1)

        ctx_from = eval(ctx_from)
        ctx_to = eval(ctx_to)

        idx_from = int(idx_from)
        idx_to = int(idx_to)

        f_word = ctx_from[idx_from]
        t_word = ctx_to[idx_to]

        print(f"{f_word} : {t_word}")

        # if (f_word, t_word) not in positive_pairs:
        #     print(line)
