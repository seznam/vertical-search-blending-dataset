#!/usr/bin/awk -f

#FILE         gen_pos_metrics.awk
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.

BEGIN{
    FS = "\t"
}
{
    split($6,avail_hints," ",spl)
    reward=$9
    prop=$10
    if(prop<0.0001){prop=0.00001}
    action=$11
    pos=$12
    click=(reward>0);
    happy_click=(reward==2)
    vertical_click=(action>0)*click
    happy_vert_click=(action>0)*happy_click
    shown_vert=action>0
    num_action=1+length(avail_hints)

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.5f\t%d\n", click, happy_click, vertical_click, happy_vert_click, shown_vert, num_action, action, prop, pos)
}
END{
    printf("click\thappy_click\tvertical_click\thappy_vert_click\tshown_vert\tnum_actions\taction\tprop\tpos")  > "pos_header.tsv"; close("pos_header.tsv")
}
