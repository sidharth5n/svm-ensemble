% add -largeArrayDims on 64-bit machines

mex COMPFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
mex COMPFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
mex COMPFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims libsvmtrain.c ../libsvm/svm.cpp svm_model_matlab.c
%mex COMPFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims svmpredict.c ../libsvm/svm.cpp svm_model_matlab.c

%mex -largeArrayDims CXX=g++ -O -c svm.cpp 
%mex -largeArrayDims -O svm_model_matlab.c
%mex -largeArrayDims -lstdc++ -O -c svm.cpp 
%mex -largeArrayDims -O -c svm.cpp 
%mex -largeArrayDims -O -c svm_model_matlab.c
%%%%%%Used originally
%%%%%%%%%%mex -largeArrayDims -O libsvmtrain.c svm.o svm_model_matlab.o

%mex -largeArrayDims CXX=g++ -I./ libsvmtrain.cpp svm.obj svm_model_matlab.obj

%%%%%%%%%%%%%%%%These files are not used - originally
%mex -O libsvmtrain.c svm.obj svm_model_matlab.obj %%%%%added line
%mex -O svmpredict.c svm.obj svm_model_matlab.obj
%mex -O libsvmread.c
%mex -O libsvmwrite.c
