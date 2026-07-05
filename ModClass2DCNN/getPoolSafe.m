function pool = getPoolSafe()
%GETPOOLSAFE Get a parallel pool if Parallel Computing Toolbox is available.
%   POOL = GETPOOLSAFE() returns the current parallel pool, POOL, if the
%   Parallel Computing Toolbox is installed and licensed. If no pool
%   exists yet, one is started. If the toolbox is not available, POOL is
%   returned as an empty array so calling code can run without it.

if exist("gcp", "file") && license("test", "distrib_computing_toolbox")
    pool = gcp;
    if isempty(pool)
        pool = parpool;
    end
else
    pool = [];
end

end
