#!/usr/bin/awk -f

#FILE         vowpal_line_cb_test.awk
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.
# prepare vowpal wabbit --cb compatible input

BEGIN{
    FS = "\t"
}
{
    split($5, a, "-", sps);
    split($6, hints, " ", spt);
    printf("1 ")
    for(i in hints){
        printf("%d ", hints[i]+1)
    }
    if($11==0) {
        printf("| q=%d nt:%d ns=%d hw=%s wd=%d h=%d d=%d p:%d\n", $2, $3, $4, $7, (a[3] + 31*a[2] - 2)%7, a[4], $8, $12)
    }
    if($11>0) {
        printf("| q=%d nt:%d ns=%d hw=%s wd=%d h=%d p:%d\n", $2, $3, $4, $7, (a[3] + 31*a[2] - 2)%7, a[4], $12)
    }
}
