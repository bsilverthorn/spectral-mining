import numpy 
import csv
import specmine
import scipy.sparse
import scipy.sparse.linalg
#import matplotlib.pyplot as plt
import time

def main():
    ''' silly script for sizing max eigenvalue computation that arpack can handle'''

    sample_min = 100
    sample_max = 1000
    sample_step = 50

    ev_min = 2
    ev_max = 50
    ev_step = 11
   
    w = csv.writer(file(specmine.util.static_path( \
        'ev_test0.csv'),'wb'))
    w.writerow(['num_samples','num_evs','run_time'])
    #R = numpy.zeros( ( len(range(sample_min,sample_max,sample_step)), \
    #                   len(range(ev_min,ev_max,ev_step))  ) )
    A = numpy.random.standard_normal((1000,1000))
    A = numpy.dot(A.T,A) # make symmetric
    A[A<0.5] = 0 # make it sparse
    t1 = time.time()
    (lam,vec) = scipy.sparse.linalg.eigsh(A, 10, which = 'SM')
    t2 = time.time()
    dt = t2-t1
    print 'time to do 10 evs on 1000 samples: ', dt

    B = A[:100,:100] 
    t1 = time.time()
    (lam,vec) = scipy.sparse.linalg.eigsh(B, 10, which = 'SM')
    t2 = time.time()
    dt = t2-t1
    print 'time to do 10 evs on 100 samples: ', dt


    for i in xrange(sample_min,sample_max,sample_step):
        for j in xrange(ev_min,ev_max,ev_step):
            print 'using ', i, ' samples and ', j, 'eigenvalues'
            A = numpy.random.standard_normal((i,i))
            A = numpy.dot(A.T,A) # make symmetric
            A[A<0.5] = 0 # make it sparse
            A = scipy.sparse.csr_matrix(A)
            t1 = time.time()
            (lam,vec) = scipy.sparse.linalg.eigsh(A, j, which = 'SM')
            t2 = time.time()
            dt = t2-t1
            #R[i,j] = dt
            w.writerow([i,j,dt])
    
    #plt.imshow(R)
    #plt.show()
    
        
    



if __name__ == "__main__":
    main()

