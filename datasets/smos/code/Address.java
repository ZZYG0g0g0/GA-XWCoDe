package smos.bean;
import java.io.Serializable;

/**
 *  Class used to model an address.
 *
 * 
 */
public class Address implements Serializable{
	
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -9194626030239503689L;
	
	private int idAddress;
	private String name;
	
	/**
	 * The class builder.
	 */
		public Address(){
		this.idAddress= 0;
	}
		
	/**
	 * @return address id.
	 */
		
	public int getIdAddress() {
		return idAddress;
	}
	
	/**
	 * Set the address id.
	 * @param pIdAddress
	 * 			the id to set.
	 */
	public void setIdAddress(int pIdAddress) {
		this.idAddress = pIdAddress;
	}
	
	/**
	 * @return the address name.
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * Set address name.
	 * @param pName
	 * 			Name to set.
	 */
	public void setName(String pName) {
		this.name = pName;
	}
	
	
}
