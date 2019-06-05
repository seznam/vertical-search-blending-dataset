#!/usr/bin/awk -f

#FILE         vowpal_line_cb.awk
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.
# prepare vowpal wabbit --cb compatible input

BEGIN{
    FS = "\t"
}
{
    split($5, a, "-", sps); # time
    split($6, hints, " ", spt); # available verticals
    if($9>0){
        loss=0.00001
    }
    if($9 ==0) {
        loss=0.99999
    }
    prop=$10
    if($10<0.0001){prop=0.00001}
    if($11!=0) {
        printf("1 ")
    }
    if($11==0) {
        printf("1:%.5f:%.5f ", loss, prop)
    }
    for(i in hints){
        if(hints[i]==$11){
            printf("%d:%.5f:%.5f ", $11+1, loss, prop)
        }
        if(hints[i]!=$11) {
            printf("%d ", hints[i]+1)
        };
    }
    if($11==0) {
        printf("| q=%d nt:%d ns=%d hw=%s wd=%d h=%d d=%d p:%d\n", $2, $3, $4, $7, (a[3] + 31*a[2] - 2)%7, a[4], $8, $12)
    }
    if($11>0) {
        printf("| q=%d nt:%d ns=%d hw=%s wd=%d h=%d p:%d\n", $2, $3, $4, $7, (a[3] + 31*a[2] - 2)%7, a[4], $12)
    }
}
