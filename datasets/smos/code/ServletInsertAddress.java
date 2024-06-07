package smos.application.addressManagement;

import java.io.IOException;
import java.sql.SQLException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpSession;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import smos.Environment;
import smos.bean.User;
import smos.bean.Address;
import smos.exception.DuplicatedEntityException;
import smos.exception.EntityNotFoundException;
import smos.exception.InvalidValueException;
import smos.exception.MandatoryFieldException;
import smos.storage.ManagerAddress;
import smos.storage.ManagerUser;
import smos.storage.connectionManagement.exception.ConnectionException;

/**
 * Servlet used to enter an address in the database
 * 
 * @author Vecchione Giuseppe
 */
public class ServletInsertAddress extends HttpServlet {

	private static final long serialVersionUID = 8318905833953187814L;
	
	/**
	 * Definition of the doGet method
	 * 
	 * @param pRequest
	 * 
	 * @param pResponse
	 * 
	 */
	
	public void doGet(HttpServletRequest pRequest, HttpServletResponse pResponse){
		String gotoPage="./showAddressList";
		String errorMessage="";
		HttpSession session = pRequest.getSession();
		ManagerUser managerUser= ManagerUser.getInstance();
		ManagerAddress managerAddress= ManagerAddress.getInstance();
		User loggedUser = (User) session.getAttribute("loggedUser");
		try {
				if(loggedUser==null){
					pResponse.sendRedirect("./index.htm");
					return;
				}
				if(!managerUser.isAdministrator(loggedUser)){
					errorMessage= "The logged-in user does not have access to the feature!";
					gotoPage="./error.jsp";
				}
				
				Address address= new Address();
				address.setName(pRequest.getParameter("name"));
				
				/**
				 *Verify that the address is not present in the database
				 * and we insert it
				 */
				if(!managerAddress.exists(address)){
					managerAddress.insert(address);
				}else{
					throw new DuplicatedEntityException("Existing address");
				}
				
			} catch (IOException ioException) {
				errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
				gotoPage = "./error.jsp";
				ioException.printStackTrace();
			} catch (SQLException sqlException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + sqlException.getMessage();
				gotoPage = "./error.jsp";
				sqlException.printStackTrace();
			} catch (EntityNotFoundException entityNotFoundException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + entityNotFoundException.getMessage();
				gotoPage = "./error.jsp";
				entityNotFoundException.printStackTrace();
			} catch (ConnectionException connectionException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + connectionException.getMessage();
				gotoPage = "./error.jsp";
				connectionException.printStackTrace();
			} catch (MandatoryFieldException mandatoryFieldException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + mandatoryFieldException.getMessage();
				gotoPage = "./error.jsp";
				mandatoryFieldException.printStackTrace();
			} catch (InvalidValueException invalidValueException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + invalidValueException.getMessage();
				gotoPage = "./error.jsp";
				invalidValueException.printStackTrace();
			} catch (DuplicatedEntityException duplicatedEntityException) {
				errorMessage =  Environment.DEFAULT_ERROR_MESSAGE + duplicatedEntityException.getMessage();
				gotoPage = "./error.jsp";
				duplicatedEntityException.printStackTrace();
			}
			session.setAttribute("errorMessage", errorMessage);
			try {
				pResponse.sendRedirect(gotoPage);
			} catch (IOException ioException) {
				errorMessage = Environment.DEFAULT_ERROR_MESSAGE + ioException.getMessage();
				gotoPage = "./error.jsp";
				ioException.printStackTrace();
			}
		}
	
	
	/**
	 *Definition of the doPost method
	 * 
	 * @param pRequest
	 * @param pResponse
	 * 
	 */
	
	protected void doPost(HttpServletRequest pRequest, 
			HttpServletResponse pResponse) {
		this.doGet(pRequest, pResponse);
	}
	

}
