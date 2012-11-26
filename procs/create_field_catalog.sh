#Combines all diferent catalogs for different pointings and quadrants
#in the given field. Some targets can have been observed more than
#once and so a refinement must be done.

Field=$1 
prefix=~/${Field}_new/reduction/mos/${Field}
echo ${prefix}

cat \
${prefix}_p1/Q1/catalog.txt ${prefix}_p1/Q2/catalog.txt \
${prefix}_p1/Q3/catalog.txt ${prefix}_p1/Q4/catalog.txt \
${prefix}_p2a/Q1/catalog.txt ${prefix}_p2a/Q2/catalog.txt \
${prefix}_p2a/Q3/catalog.txt ${prefix}_p2a/Q4/catalog.txt \
${prefix}_p2b/Q1/catalog.txt ${prefix}_p2b/Q2/catalog.txt \
${prefix}_p2b/Q3/catalog.txt ${prefix}_p2b/Q4/catalog.txt \
${prefix}_p3/Q1/catalog.txt ${prefix}_p3/Q2/catalog.txt \
${prefix}_p3/Q3/catalog.txt ${prefix}_p3/Q4/catalog.txt \
> ~/catalogs/${Field}/catalog_${Field}_total.txt

head -n 1 ~/catalogs/${Field}/catalog_${Field}_total.txt > ~/catalogs/${Field}/aux1.remove
grep -v 'OBJECT' ~/catalogs/${Field}/catalog_${Field}_total.txt > ~/catalogs/${Field}/aux2.remove
rm -f ~/catalogs/${Field}/catalog_${Field}_total.txt
cat ~/catalogs/${Field}/aux1.remove ~/catalogs/${Field}/aux2.remove > ~/catalogs/${Field}/catalog_${Field}_total.txt
rm -f ~/catalogs/${Field}/*.remove