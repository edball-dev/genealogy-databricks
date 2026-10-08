# Census age mismatches: review outcome

**Update:** the age gate was then raised from 5 to 10 years (notebook_01 Cells 5e-5h3, notebook_02 `AGE_CHECK_TOLERANCE`, DQ-025). Of the 17 rows below, 9 now pass the gate (the six "same person" rows with a gap of 6-9 plus the three other same-person rows within 10). DQ-025 `LINK_AGE_MISMATCH` is now 8 rows: the six different-person rows with a gap over 10 (Mary Ann Balls, Elizabeth Balls, Mary Ann Eastoe, Herbert Davis, Ann Pearson, Mary Pearson) and the two doubtful ones (Thomas Cope, Susan Ambrose). The Israel/Sarah Clifford link has a gap of exactly 10, so DQ-025 does not flag it; the rebuild drops it because the names no longer match.

DQ-025 `LINK_AGE_MISMATCH` listed 41 rows. Two were corrected earlier (Pearson 1881 42→62, Ballantyne 1901 51→57); the other 39 were checked against the images in batches. Result:

- **22 mention rows corrected** (transcript line and `silver_transcript_person_mention` age, marked `COMPLETE` with a correction note): the age had been misread, and the corrected age agrees with the tree.
- **1 name corrected:** Clifford 1841, "Sarah do" is "Israel Clifford (male)", age 12 (mention renamed, role Son).
- **17 rows still flagged**, all read correctly from the page.

## Remaining 17

### Different person from the tree match (7): the rebuild should drop these links

| Page | Page person (age) | Linked tree person (b.) |
|---|---|---|
| BALLS_Mary_1871_Census-p2 | Mary Ann Balls (18) | Mary Smith (1789) |
| BALLS_Mary_1871_Census-p2 | Elizabeth Balls (5) | Elizabeth Balls (1826) |
| EASTER_Mary Ann_1851_Census p1 | Mary Ann Eastoe (40) | Mary Ann Easter (1848) |
| DAVIS_Joseph (Sr)_1891_Census | Herbert Davis (6) | Herbert Davis (1866) |
| PEARSON_John_1841_Census | Ann Pearson (7) | Ann Pearson (1799) |
| PEARSON_John_1841_Census | Mary Pearson (20) | Mary Pearson (1806) |
| CLIFFORD_Edward_1841_Census | Israel Clifford (12) | Sarah Clifford (1819) |

### Same person, age on the page differs from the tree (10): inside the new 10-year gate except the last two

| Page | Page age → implied birth | Tree birth | Gap | Other censuses agree with the tree? |
|---|---|---|---|---|
| EASTOE_Charles_1881_Census | 38 → 1843 | 1837 | 6 | yes (1839-41) |
| GIRDLESTONE_Samuel_1861_Census (Martha) | 30 → 1831 | 1825 | 6 | yes (1826-29) |
| GIRDLESTONE_Samuel_1871_Census (Martha) | 38 → 1833 | 1825 | 8 | yes |
| HALLAM_Samuel_1871_Census | 47 → 1824 | 1830 | 6 | yes (1830-31) |
| MCLEAN_Thomas_1841_Census | 44 → 1797 | 1791 | 6 | yes (1790) |
| SHEARER_William_1841_Census | 20 → 1821 | 1815 | 6 | no other census |
| BALLS_Charles_1881_Census | 27 → 1854 | 1846 | 8 | mixed (1842-1855) |
| BALLS_Charles_1901_Census | 46 → 1855 | 1846 | 9 | mixed |
| COPE_Thomas_1841_Census | 40 → 1801 | 1785 | 16 | yes (1787) |
| AMBROSE_Susan_1881_ Census-Absent | 85 → 1796 | 1820 | 24 | yes (1816-21) |

The last three are doubtful and may need the tree birth year or the person checked.
