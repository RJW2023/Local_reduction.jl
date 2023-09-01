#This is a solver for semi-infinite programming (SIP) using local reduction 
#method programmed in Julia and is embedded on JuMP.

#The general form of SIP is:
#       min f(x), x in X 
#  s.t. g_i(x,y_i) <= 0 for all y_i in Y_i, i = 1,...,N

#To use this function the following input arguments are needed:
# 1. User-defined objective and constraint functions of your SIP problem, 
# e.g. SIP_obj = f and SIP_cons = [g_1, ..., g_n]
# 2. Lower and upper bounds of your variable x and y
# 3. A set of initial guess for y

function local_reduction_pr(SIP_obj::Function, SIP_cons::Vector, xL::Vector{Float64}, xU::Vector{Float64}, yL::Vector{Vector{Float64}}, yU::Vector{Vector{Float64}}, y_k::Vector{Vector{Float64}}; solver::Module=Ipopt, y_cons::Vector=[[]], abs_tolerance::Float64=1e-5, max_iter::Int64=200, process_storage::Bool=false)
   
    n_cons = length(SIP_cons) #Number of SIP constranits
    x_size = length(xU) #dimension of x
    y_size = zeros(Int64, n_cons) #Vector of the dimention of variable y in each consraint 
    for i = 1:n_cons
        y_size[i] = length(y_k[i])
    end

    #Define Min problem
    model_min = Model(solver.Optimizer) #create model for Min problem 
    @variable(model_min, xL[i] <= x[i=1:x_size] <= xU[i])
    @objective(model_min, Min, SIP_obj(x))

    #Define Max problem
    model_max = Array{Model}(undef, n_cons) #Create a vector to sotre models of Max prob 
    for i = 1:n_cons 
        model_max[i] = Model(solver.Optimizer) #Create models for each constraint
        @variable(model_max[i], yL[i][n] <= y[n=1:y_size[i]] <= yU[i][n])

        if y_cons != [[]] #If exist, add constraint of y for Max problems
            if y_cons[i] != []
                for n = 1:length(y_cons[i])
                    @constraint(model_max[i], y_cons[i][n](y) <= 0)
                end
            end
        end
    end

    #Solving Min and Max problems
    k = 0 # Initialize the number of iterations
    x_k = zeros(x_size) #Initialize the solution of Min prob in kth iteration
    check_value = zeros(n_cons) #value of g(x_k, y_k+1)
    check_result = true # Condition checked at the end of each itera

    #If need, store intermediate data
    if process_storage == true
        x_storage = zeros(x_size, max_iter)
        y_storage = [zeros(y_size[i], max_iter) for i in 1:n_cons]
    end

    while k == 0 || (~check_result) #If in fisrt itera or conditions aren't satisfied, loop continues
        if k >= max_iter
            println()
            println("Warning! Exceed the maximum allowable number of iterations: $max_iter")
            if process_storage == true
                return (x_storage, y_storage)
            end
            exit()
        end

        k = k + 1

        #Solving Min problems
        if k == 1 # In 1st iteration, add all constraints into min prob model 
            for (i,f) in enumerate(SIP_cons)
                if length(y_k[i]) == 1
                    @constraint(model_min, f(x, y_k[i][1]) <= 0)
                else
                    @constraint(model_min, f(x, y_k[i]) <= 0)
                end
            end
        else #Add most violated constraint in the coming iteration
            most_vio = findmax(check_value)[2] #find worst scenario
            if length(y_k[most_vio]) == 1
                @constraint(model_min, SIP_cons[most_vio](x, y_k[most_vio][1]) <= 0)
            else
                @constraint(model_min, SIP_cons[most_vio](x, y_k[most_vio]) <= 0)
            end
        end

        optimize!(model_min)
        x_k = value.(x)
        if process_storage == true
            x_storage[:,k] = x_k
        end

        #Solving Max problems
        check_result = true # Initialize condition checking result at each itera
        for (i, f) in enumerate(SIP_cons)
            y_ref = Array{VariableRef}(undef,y_size[i]) #Array to store VariableRef of each Max prob model
            for n = 1:y_size[i]
                y_ref[n] = variable_by_name(model_max[i], "y[$(n)]")
            end
            @objective(model_max[i], Max, f(x_k, y_ref))

            optimize!(model_max[i])    
            y_Q = value.(y_ref)
            check_value[i] = f(x_k, y_Q)  
            if process_storage == true
                y_storage[i][:,k] = y_Q
            end
            y_k[i] = y_Q 
            check_result = check_result && (check_value[i] <= abs_tolerance) #If g(x_k, y_k+1) <=0 are satisfied for all Max probs, loop ends
        end
    end

#Output results
    println()
    println("The number of iterations: $k")
    println("Solution of Y is: $y_k")
    println("Solution of X is: $x_k")
    println()
    if process_storage == true
        return(x_storage, y_storage)
    end
end
