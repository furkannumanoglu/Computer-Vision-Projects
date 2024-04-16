FURKAN NUMANOĞLU
2448710

Cv2 and numpy imported

Make sure that the main.py file is in the dataset folder and that query_1, query_2, query_3, support_96 and instanceNames.txt files are in the dataset folder.

For start the program:

main(histogramType, binNumber, grid, query, isHSV)

histogramType: "perChannel" or "3D"
binNumber: 256 / quantization_interval
binNumber should be integer
Grid: 1,4,9,16.  (If you want 4x4 grid structure, you should write 16, it should be integer)
Query: 1,2,3 (in "query_x" program wants to get that x. It should be integer)
isHSV = bool (True or False)

Program will print accuracy as percent. 

e.g if you call : 
main("perChannel", 4, 4, 1, False)
Program print: "99%"


main("3D", 4, 1, 3, True)
Program prints: "17.5%"