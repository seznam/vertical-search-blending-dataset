#!/usr/bin/awk -f

#FILE         expand_position.awk
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.

BEGIN{
    FS = "\t"
}
{
    split($6,avail, " ", seps);
    hint_sep = 10
    for(pos=0;pos<14;pos++){
        dom=$(11+pos*4); rev=$(8+pos*4); prop=$(9+pos*4); action=$(10+pos*4);
        if(action>0){
            dom=$(11+(pos+1)*4)
            if(hint_sep < 3){
                printf("Error skip\tid:%s\tpos:%d\n", $1, pos) >> "error.log"
                close("error.log")
                break
            }
        };
        if(length(dom)<1){break}
        printf("%s\t%s\t%d\t%d\t%s\t", $1, $2, $3, $4, $5);
        if(hint_sep >= 3){
            for(i in avail){printf("%d ", avail[i])};
        }
        if(hint_sep < 3){
            printf(" ")
        }
        printf("\t%s\t%s\t%d\t%.5f\t%d\t%d\n", $7,dom,rev,prop, action, pos); 
        if(action>0){
            for(i in avail){
                if(avail[i] == action){
                    delete avail[i];
                    break;
                }
            }

            hint_sep = 0
        }
        if(action==0){
            hint_sep += 1
        }
    }
}
END{
    printf("req_id\tquery\tnum_tokens\tnum_skips\ttimestamp\tavail_verts\thw\tdomain\treward\tprop\taction\tposition") > "header_position.tsv"; close("header_position.tsv")
}
