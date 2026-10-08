# Census age mismatches to review before the Cell 5 rebuild

39 census links (HIGH/MEDIUM) where the age on the page implies a birth year more than 5 years from the tree's (DQ-025 `LINK_AGE_MISMATCH`). Luke Pearson 1881 (42 → 62) and John Ballantyne 1901 (51 → 57) are already corrected and are not listed.

For each row, check the age on the image. A rebuild applies the same 5-year gate, so a correct link with a misread age would be dropped, while a wrong-person link is dropped correctly.

- **Misread age:** correct the transcript and the mention row (as for Cuthbertson, Pearson and Ballantyne).
- **Wrong person:** no data change needed, the rebuild drops the link.
- **Tree birth year wrong:** fix the tree.

"Expected" is document year minus the tree birth year. "Other censuses" are the same person's other census links as `year:age→implied birth`.

## A. Mention looks like a different person (12)

The page person differs by a generation, a name, or a role from the tree person, and in most rows the other censuses support the tree.

| Tree person (b.) | Page | Page age → birth | Expected | Other censuses |
|---|---|---|---|---|
| Mary Smith (1789) | BALLS_Mary_1871_Census-p2 (mention "Mary Ann Balls", Daughter) | 18 → 1853 | 82 | 1871:78→1793, 1861:70→1791 |
| Elizabeth Balls (1826) | BALLS_Mary_1871_Census-p2 | 5 → 1866 | 45 | 1861:35→1826 |
| Mary Ann Easter (1848) | EASTER_Mary Ann_1851_Census p1 (mention "Mary Ann Eastoe", Head) | 40 → 1811 | 3 | 1851:3→1848 |
| Herbert Davis (1866) | DAVIS_Joseph (Sr)_1891_Census | 6 → 1885 | 25 | 1871:4→1867 |
| John William Balls (1820) | BALLS_John William_1871_Census | 32 → 1839 | 51 | 1851:30→1821, 1881:40→1841 |
| John William Balls (1820) | BALLS_John William_1881_Census | 40 → 1841 | 61 | 1851:30→1821, 1871:32→1839 |
| Lucy Lamb (1804) | PRATT_Lucy_1871_Census (mention "Lucy Youell", Head) | 46 → 1825 | 67 | 1841:37→1804, 1851:47→1804 |
| Samuel Hallam (1830) | HALLAM_Samuel_1881_Census (Head) | 29 → 1852 | 51 | 1841:11, 1851:20, 1861:31→1830-31; 1871:47→1824 |
| William Easter (1877) | EASTER_William_1891_Census (Nephew) | 23 → 1868 | 14 | 1881:3, 1901:23, 1911:34, 1921:43→1877-78 |
| Ann Pearson (1799) | PEARSON_John_1841_Census (Child) | 7 → 1834 | 42 | none |
| Mary Pearson (1806) | PEARSON_John_1841_Census (Child) | 20 → 1821 | 35 | none |
| Sarah Clifford (1819) | CLIFFORD_Edward_1841_Census (Daughter) | 12 → 1829 | 22 | none |

The two John William Balls pages agree with each other (1839, 1841) but not with the 1851 page, which agrees with the tree. That points to a second, younger John William Balls, not a misread.

## B. Same name, other censuses support the tree (25)

The age on the page is the odd one out, so a misread is likely. Look at the digit first.

| Tree person (b.) | Page | Page age → birth | Expected | Other censuses |
|---|---|---|---|---|
| Ann Cooper (1786) | WARRINGTON_Robert_1851_Census | 55 → 1796 | 65 | 1841:54→1787 |
| Ann Warrington (1823) | WARRINGTON_Robert_1851_Census | 22 → 1829 | 28 | 1841:20→1821 |
| Charles Ball (1846) | BALLS_Charles_1881_Census | 27 → 1854 | 35 | 1851:4→1847, 1891:49→1842, 1901:46→1855, 1911:64→1847 |
| Charles Ball (1846) | BALLS_Charles_1901_Census | 46 → 1855 | 55 | as above |
| Charles Eastoe (1837) | EASTOE_Charles_1881_Census | 38 → 1843 | 44 | 1871:30→1841, 1891:50→1841, 1911:72→1839, 1921:82→1839 |
| Edward Cope (1835) | COPE_Edward_1881_Census | 54 → 1827 | 46 | 1841:7, 1851:17→1834; 1871:36, 1891:56, 1911:76→1835; 1901:65→1836 |
| Elizabeth Moore (1825) | THORPE_William_1851_Census (mention "Elizabeth Thorpe", Wife) | 20 → 1831 | 26 | 1861:35, 1871:45, 1881:55→1826 |
| Elizabeth Siddon (1827) | SIDDON_Elizabeth_1841_Census | 44 → 1797 | 14 | 1871:44→1827, 1881:53→1828 |
| Elizabeth Siddon (1827) | CLIFFORD_William_1851_Census (mention "Elizabeth Clifford", Wife) | 34 → 1817 | 24 | 1871:44→1827, 1881:53→1828 |
| Ellen Jane Pearson (1870) | PEARSON_Luke_1891_Census (mention "Ellen Pearson", Daughter) | 34 → 1857 | 21 | 1871:0, 1881:10→1871 |
| George Carr Jessop (1834) | JESSOP_George Carr_1891_Census | 51 → 1840 | 57 | 1861:27, 1881:47, 1901:67→1834; 1871:39→1832 |
| Henry Cope (1875) | COPE_Edward_1901_Census | 35 → 1866 | 26 | 1881:6→1875, 1911:35, 1921:45→1876 |
| James Henry Girdlestone (1863) | GIRDLESTONE_Samuel_1891_Census | 17 → 1874 | 28 | 1871:8→1863, 1881:17→1864, 1901:38→1863 |
| Martha Miller Isbill (1825) | GIRDLESTONE_Samuel_1861_Census | 30 → 1831 | 36 | 1851:22→1829, 1881:55→1826, 1901:74→1827 |
| Martha Miller Isbill (1825) | GIRDLESTONE_Samuel_1871_Census | 38 → 1833 | 46 | as above |
| Martha Miller Isbill (1825) | GIRDLESTONE_Samuel_1891_Census | 44 → 1847 | 66 | as above |
| Reuben Piggin (1854) | PIGGIN_Reuben Rogers_1881_Census | 21 → 1860 | 27 | 1871:14→1857 |
| Samuel Hallam (1830) | HALLAM_Samuel_1871_Census | 47 → 1824 | 41 | 1841:11, 1851:20, 1861:31→1830-31 |
| Sarah Ann Ambrose (1856) | BALLS_Charles_1891_Census (mention "Sarah Ball", Wife) | 46 → 1845 | 35 | 1861:5, 1871:14, 1881:23, 1901:45→1856-58; 1911:57, 1921:67→1854 |
| Sarah West (1807) | THORPE_Thomas_1871_Census (mention "Sarah Thorpe", Wife) | 73 → 1798 | 64 | 1841:32→1809, 1851:47→1804, 1861:54→1807 |
| Susan Payne (1820) | AMBROSE_Susan_1881_ Census-Absent (mention "Susan Ambrose", Head) | 85 → 1796 | 61 | 1851:30, 1861:41→1820-21; 1871:55, 1891:75→1816 |
| Thomas Cope (1785) | COPE_Thomas_1841_Census | 40 → 1801 | 56 | 1851:64→1787 |
| Thomas McLean (1791) | MCLEAN_Thomas_1841_Census | 44 → 1797 | 50 | 1851:61→1790 |
| Thomas Palmer (1830) | PALMER_Thomas_1891_Census | 54 → 1837 | 61 | 1861:30, 1881:50→1831; 1871:38→1833 |
| William Clifford (1822) | CLIFFORD_William_1851_Census | 36 → 1815 | 29 | 1841:15→1826, 1871:48→1823, 1881:60→1821, 1891:70→1821 |

The three Martha Isbill pages (30, 38, 44) disagree with each other as well as with the tree, which suggests misreads or a different Martha Girdlestone on those pages. Ann Cooper (55 for 65) and Ann Warrington (22 for 28) differ by one digit from the expected age.

## C. Nothing else to compare (2)

| Tree person (b.) | Page | Page age → birth | Expected | Other censuses |
|---|---|---|---|---|
| Gertrude May Easter (1916) | EASTER_William_1921_Census (Daughter) | 12 → 1909 | 5 | none |
| William Shearer (1815) | SHEARER_William_1841_Census (Head) | 20 → 1821 | 26 | none |
