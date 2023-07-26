
function local_reduction_1(y_k::Vector{Vector{Float64}}; SIP_obj_L="empty", SIP_obj_NL="empty", SIP_cons_L_vec::Vector=["empty"], SIP_cons_NL_vec::Vector=["empty"], xL="infinity", xU="infinity", yL="infinity", yU="infinity", abs_tolerance::Number=1e-5)
    #Define Min problem
    model_min = Model(Ipopt.Optimizer) #create Min problem model
    if xL == "infinity"
        x_size = length(xU)
        if x_size == 1
            @variable(model_min, x <= xU)
        else
            @variable(model_min, x[i=1:x_size] <= xU[i])
        end
    elseif xU == "infinity"
        x_size = length(xL)
        if x_size == 1
            @variable(model_min, x >= xL)
        else    
            @variable(model_min, x[i=1:x_size] >= xL[i])
        end
    else
        x_size = length(xU)
        if x_size == 1
            @variable(model_min, xL <= x <= xU)
        else    
            @variable(model_min, xL[i] <= x[i=1:x_size] <= xU[i])
        end
    end
    if SIP_obj_NL == "empty" 
        @objective(model_min, Min, SIP_obj_L(x))
    else 
        register(model_min, :max_obj, 1, SIP_obj_NL; autodiff = true)
        @NLobjective(model_min, Min, max_obj(x))
    end


    #Define Max problem
    if SIP_cons_L_vec == ["empty"]
        n_L = 0 #Number of linear constraints
    else
        n_L = length(SIP_cons_L_vec)
    end
    if SIP_cons_NL_vec == ["empty"]
        n_NL = 0 #Number of NLconstraints
    else
        n_NL = length(SIP_cons_NL_vec)
    end

    y_size = zeros(n_L+n_NL)#Array of dimention of variable y in ith consraint 
    for i = 1:n_L+n_NL
        y_size[i] = length(y_k[i])
    end

    model_max = Array{Model}(undef, n_L+n_NL)
    
    for i = 1:n_L+n_NL #Add models for constraints
        model_max[i] = Model(Ipopt.Optimizer)
        if yL == "infinity"
            if y_size[i] == 1 
                @variable(model_max[i], y <= yU[i][1])
            else
                @variable(model_max[i], y[n=1:Int64(y_size[i])] <= yU[i][n])
            end
        elseif yU == "infinity"
            if y_size[i] == 1 
                @variable(model_max[i], y >= yL[i][1])
            else
                @variable(model_max[i], y[n=1:Int64(y_size[i])] >= yL[i][n])
            end
        else
            if y_size[i] == 1 
                @variable(model_max[i], yL[i][1] <= y <= yU[i][1])
            else
                @variable(model_max[i], yL[i][n] <= y[n=1:Int64(y_size[i])] <= yU[i][n])
            end
        end
    end


    #Solving Min and Max problems
    k = 0 # Initialize the number of iteration
    x_k = zeros(1,x_size)
    check_value = zeros(1,n_L+n_NL) #result of f(x_k, y_k+1)
    check_result = true 
    
    while k == 0 || (~check_result)
        k = k + 1
        #Solving Min problems
        if k == 1 # In 1st iteration, add all constraints into model P
            if n_L != 0
                for (i,f) in enumerate(SIP_cons_L_vec)
                    if length(y_k[i]) == 1
                        cons_index = @expression(model_min, f(x, y_k[i][1]))
                    else
                        cons_index = @expression(model_min, f(x, y_k[i]))
                    end
                    @constraint(model_min, cons_index <= 0)
                end
            end
            if n_NL != 0
                for (i,f) in enumerate(SIP_cons_NL_vec)
                    f_sym = Symbol("f_$(k)_$(i)")
                    register(model_min, f_sym, 2, f; autodiff = true)
                    if length(y_k[n_L+i]) == 1
                        add_nonlinear_constraint(model_min, :($(f_sym)($(x), $(y_k[n_L+i][1])) <= 0))
                    else
                        add_nonlinear_constraint(model_min, :($(f_sym)($(x), $(y_k[n_L+i])) <= 0))
                    end
                end
            end
        else #Add most violated constraint in the coming iteration
            most_vio = findmax(check_value)[2][2]
            if most_vio <= n_L
                if length(y_k[most_vio]) == 1
                    @constraint(model_min, SIP_cons_L_vec[most_vio](x, y_k[most_vio][1]) <= 0)
                else
                    @constraint(model_min, SIP_cons_L_vec[most_vio](x, y_k[most_vio]) <= 0)
                end
            else
                f_sym = Symbol("f_$(k)")
                register(model_min, f_sym, 2, SIP_cons_NL_vec[most_vio-n_L]; autodiff = true)
                if length(y_k[most_vio]) == 1 
                    add_nonlinear_constraint(model_min, :($(f_sym)($(x), $(y_k[most_vio][1])) <= 0))
                else
                    add_nonlinear_constraint(model_min, :($(f_sym)($(x), $(y_k[most_vio])) <= 0))
                end
            end
        end

        optimize!(model_min)

        if x_size == 1
            x_k = value(x)
            #x_k_storage[k] = x_k 
        else
            x_k = value.(x)
            #x_k_storage[k,:] = x_k 
        end
       
        #Solving Max problems
        check_result = true

        if n_L != 0
            for (i, f) in enumerate(SIP_cons_L_vec)
                y_ref = Array{VariableRef}(undef,Int64(y_size[i]))

                if y_size[i] == 1
                    y_ref = variable_by_name(model_max[i], "y")
                else
                    for n = 1:Int64(y_size[i])
                        y_ref[n] = variable_by_name(model_max[i], "y[$(n)]")
                    end
                end

                @objective(model_max[i], Max, f(x_k, y_ref))
                optimize!(model_max[i])

                if y_size[i] == 1
                    y_Q = [value(y_ref)]
                    check_value[i] = f(x_k, value(y_ref))
                else
                    y_Q = value.(y_ref)
                    check_value[i] = f(x_k, y_Q)    
                end
                y_k[i] = y_Q 
                check_result = check_result && (check_value[i] <= abs_tolerance) #if check_result is 'true' for all model Q, loop ends
            end
        end

        if n_NL != 0
            for (i, f) in enumerate(SIP_cons_NL_vec)
                index = n_L+i
                y_ref = Array{VariableRef}(undef,Int64(y_size[index]))

                if y_size[index] == 1
                    y_ref = variable_by_name(model_max[index], "y")    
                else
                    for n = 1:Int64(y_size[index])
                        y_ref[n] = variable_by_name(model_max[index], "y[$(n)]")
                    end
                end

                set_nonlinear_objective(model_max[index], MAX_SENSE, f(x_k, y_ref))
                optimize!(model_max[index])

                if y_size[index] == 1
                    y_Q = [value(y_ref)]
                    check_value[index] = f(x_k, value(y_ref))
                else
                    y_Q = value.(y_ref) 
                    check_value[index] = f(x_k, y_Q)   
                end
                y_k[index] = y_Q

                check_result = check_result && (check_value[index] <= abs_tolerance) #if check_result is 'true' for all model Q, loop ends
            end
        end
    end

    println()
    println("The number of iterations: ", k)
    println("The solution Y is ",y_k)
    println("The solution X is ",x_k)
end